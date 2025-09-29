# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean, gather_from_labels, sum_log_softmax, sum_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
import wandb
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from torch.nn.parallel import DistributedDataParallel
__all__ = ['DataParallelPPOActor']
class scale_vector(torch.nn.Module):
    def __init__(self):
        super(scale_vector, self).__init__()
        self.para = torch.nn.Parameter(torch.rand(1,1024,1))
    def forward(self, x):
        # print(">>>>>",self.para[0,0,0],self.para.is_leaf)
        return torch.exp(torch.nn.LogSoftmax(dim=-1)(x)).float() * self.para

class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
class DataParallelRPEActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        scale:nn.Module = None,
        actor_optimizer: torch.optim.Optimizer = None,
        device_mesh = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.step = 1
        # print('-------')
        if actor_optimizer is not None:
            # self.scale = scale_vector().to(torch.cuda.current_device())
            # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
            # from verl.utils.fsdp_utils import init_fn
            # sharding_strategy = ShardingStrategy.FULL_SHARD
            # self.scale = FSDP(scale_vector().to(torch.cuda.current_device()))
            # #DistributedDataParallel(scale_vector().to(torch.cuda.current_device()))
            # self.scale_optimizer = torch.optim.AdamW(self.scale.parameters(),#,#actor_module_fsdp.parameters(),#
            # lr=1e-6,
            # betas=(0.9, 0.999),
            # weight_decay=1e-2
            # )
            self.scale = scale
            # param_init_fn=init_fn,
            # use_orig_params=False,
            # device_id=torch.cuda.current_device(),
            # sharding_strategy=sharding_strategy,  # zero3
            # # sync_module_states=True,
            # device_mesh=device_mesh,
            # forward_prefetch=False)
            # print('Loaded!')
        # self.scale = scale
        self.actor_optimizer = actor_optimizer
        # if self.actor_optimizer is not None:
           
        #     self.actor_optimizer.add_param_group({"params": self.scale.parameters()})
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ref_logits=None
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.sumlog = torch.compile(verl_F.sum_log_softmax, dynamic=True)
    def _forward_micro_batch(self, micro_batch, temperature, is_ref=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                # logits_rmpad = torch.round(logits_rmpad*10) / 10
                logits_rmpad.div_(temperature)
                # q_values = torch.nn.Softmax(dim=-1)(logits_rmpad)
                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                # logp = torch.log(torch.nn.Softmax(dim=-1)(logits_rmpad.float())+1)*torch.max(logits_rmpad.float(),dim=-1,keepdim=True)[0] / 100
                # logp = logits_rmpad#torch.log(torch.nn.Softmax(dim=-1)(logits_rmpad.float())+1)
                # logp = torch.exp(torch.nn.LogSoftmax(dim=-1)(logits_rmpad.float()))
                # logp = (torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits_rmpad.float()))-1) / 1000
                # logp = torch.log(torch.nn.Softmax(dim=-1)(logits_rmpad.float())+1)
                # log_probs = gather_from_labels(logp, input_ids_rmpad_rolled)
                # # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                logit = verl_F.gather_from_labels(logits_rmpad, input_ids_rmpad_rolled)
                q_values = self.sumlog(logits_rmpad)
                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    logit = gather_outpus_and_unpad(logit, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    logits_rmpad = gather_outpus_and_unpad(logits_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    q_values= gather_outpus_and_unpad(q_values,
                                                        gather_dim=0,
                                                        unpad_dim=0,
                                                        padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                full_logit = pad_input(hidden_states=logit.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                full_logits = pad_input(hidden_states=logits_rmpad.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                full_q_values = pad_input(hidden_states=q_values.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                logit = full_logit.squeeze(-1)[:, -response_length - 1:-1]
                logits = full_logits.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                # log_probs = log_probs - torch.logsumexp(torch.nn.Softmax(dim=-1)(logits),dim=-1)
                # + torch.cat([torch.logsumexp(logits[:,:-1],dim=-1) - torch.logsumexp(logits[:,1:],dim=-1), torch.zeros_like(log_probs[:,:1])],dim=1)
                q_values = full_q_values.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                # if ref_policy:
                #     q_value = logits-ref_logits
                # q_values = (logits-logits.min(dim=-1,keepdim=True)[0])/(logits.max(dim=-1,keepdim=True)[0]-logits.min(dim=-1,keepdim=True)[0])
                # q_values = q_values * torch.exp(torch.nn.LogSoftmax(dim=-1)(logits))
                # q_values = (torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits.float()))-1) / 1000
                # log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                # logp = torch.log(torch.nn.Softmax(dim=-1)(logits.float())+1)*torch.max(logits.float(),dim=-1,keepdim=True)[0] / 100
                logp = torch.log(torch.nn.Softmax(dim=-1)(logits.float())+1)
                # logp = (torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits.float()))-1) / 1000#torch.exp(torch.nn.Softmax(dim=-1)(logits.float()))-1
                log_probs = gather_from_labels(logp, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            # next_q = torch.cat([q_values[:,1:].mean(dim=-1)*q_values.shape[-1],torch.zeros_like(q_values[:,:1,0])],dim=1)
            # q_values = torch.gather(q_values, -1, micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
            # next_q = torch.cat([q_values[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
            # return q_values, next_q, log_probs
            # if not is_ref:
            #     logits = self.actor_module.module.scale(logits)
            return logits, log_probs, entropy, q_values, logit
    def   ref_forward_micro_batch(self, ref_policy, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = ref_policy(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)
                # q_values = torch.nn.Softmax(dim=-1)(logits_rmpad)
                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    logits_rmpad = gather_outpus_and_unpad(logits_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    # q_values= gather_outpus_and_unpad(q_values,
                    #                                     gather_dim=0,
                    #                                     unpad_dim=0,
                    #                                     padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                full_logits = pad_input(hidden_states=logits_rmpad.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                # full_q_values = pad_input(hidden_states=q_values.unsqueeze(-1),
                #                            indices=indices,
                #                            batch=batch_size,
                #                            seqlen=seqlen)
                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = full_logits.squeeze(-1)[:, -response_length - 1:-1]
                # log_probs = log_probs - torch.logsumexp(torch.nn.Softmax(dim=-1)(logits),dim=-1)
                # q_values = full_q_values.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
            else:  # not using rmpad and no ulysses sp
                output = ref_policy(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                # if ref_policy:
                #     q_value = logits-ref_logits
                # q_values = (logits-logits.min(dim=-1,keepdim=True)[0])/(logits.max(dim=-1,keepdim=True)[0]-logits.min(dim=-1,keepdim=True)[0])
                # q_values = q_values * torch.exp(torch.nn.LogSoftmax(dim=-1)(logits))
                #q_values = torch.exp(torch.nn.LogSoftmax(dim=-1)(logits))
                # log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                # entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            # next_q = torch.cat([q_values[:,1:].mean(dim=-1)*q_values.shape[-1],torch.zeros_like(q_values[:,:1,0])],dim=1)
            # q_values = torch.gather(q_values, -1, micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
            # next_q = torch.cat([q_values[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
            # return q_values, next_q, log_probs
            
            return logits,log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        # grad_norm = torch.nn.utils.clip_grad_norm_([self.actor_module.parameters(),self.scale.parameters()], max_norm=self.config.grad_clip)
        # self.actor_optimizer.step()
        # grad_norm_ = torch.nn.utils.clip_grad_norm_(self.scale.parameters(), max_norm=self.config.grad_clip)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            # grad_norm_ = self.scale.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_([self.actor_module.parameters()+self.scale.parameters()], max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        # self.scale_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, is_ref=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        logits_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                logits, log_probs,_, q_values,_= self._forward_micro_batch(micro_batch, temperature=temperature, is_ref=is_ref)
                # _,_, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            # logits_lst.append((torch.log(torch.nn.Softmax(dim=-1)(logits.float())+1)*torch.max(logits.float(),dim=-1,keepdim=True)[0] / 100).sum(dim=-1))
            # logits_lst.append(torch.log(torch.topk(torch.nn.Softmax(dim=-1)(logits.float()),k=20, dim=-1)[0]+1).sum(dim=-1))
            # logits_lst.append(((torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits.float()))-1) / 1000).sum(dim=-1))
            # logits_lst.append((torch.log(torch.nn.Softmax(dim=-1)(logits.float())+1)).sum(dim=-1))
            # logits_lst.append((torch.exp(torch.nn.LogSoftmax(dim=-1)(logits.float()))).sum(dim=-1))
            
            logits_lst.append(q_values)
            # logits_lst.append(sum_mean(logits.float()))
        log_probs = torch.concat(log_probs_lst, dim=0)
        logits = torch.concat(logits_lst,dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            logits = logits[revert_indices]
        
        return log_probs, logits
    # def lambda_value(self,x):

    def update_policy(self, data: DataProto,ref=None,scale=None):
        # make sure we are in training mode
        self.actor_module.train()
        # self.scale.train()
        # if self.step>75:
        #     self.config.ppo_mini_batch_size = 192
        # self.scale.para.requires_grad_(True)
        # self.actor_module.q_head_proj.cuda()
        # self.actor_module.q_head_proj.requires_grad_(True)
        
        # scale.requires_grad_(True)
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'advantages']
        # if self.config.use_kl_loss:
        # select_keys.append('ref_log_prob')
        select_keys.append('sum_ref_logits')
        select_keys.append('old_log_probs')
        select_keys.append('old_logits')
        batch = data.select(batch_keys=select_keys).batch
        q_mean = batch['old_logits'].mean(dim=0,keepdim=True)
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()
            # self.scale_optimizer.zero_grad()
            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                # breakpoint()
                # clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                # entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                # q_values, next_q, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature,ref_policy=ref_policy)
                logits, log_prob, entropy, current_q,cur_logit = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                # ref_logits,ref_log_probs = self.ref_forward_micro_batch(ref_policy=ref,micro_batch=data, temperature=temperature)
                # if self.step%10==0:
                sum_ref_logits = data['sum_ref_logits']
                # ref_log_prob = data['ref_log_prob']
                # else:
                # if self.step>0:
                old_logits = data['old_logits']#data['sum_ref_logits']
                old_log_prob = data['old_log_probs']#data['ref_log_prob']
                # else:
                # sum_ref_logits = data['sum_ref_logits']
                # ref_log_prob = data['ref_log_prob']
                q_values = (log_prob-old_log_prob)#(log_prob - 0.9*old_log_prob-0.1*ref_log_prob) #+ log_prob - ref_log_prob_
                q_values = q_values * response_mask
                # next_q = torch.cat([(((torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits.float()))-1) / 1000).sum(dim=-1)[:,1:]-sum_ref_logits[:,1:])* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # next_q = torch.cat([((torch.nn.Softmax(dim=-1)(logits.float())).sum(dim=-1)[:,1:]-sum_ref_logits[:,1:])* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # next_q = torch.cat([((torch.log(torch.nn.Softmax(dim=-1)(logits.float())+1)*torch.max(logits.float(),dim=-1,keepdim=True)[0] / 100).sum(dim=-1)[:,1:]-sum_ref_logits[:,1:])* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # next_q = torch.cat([(((torch.exp(torch.nn.LogSoftmax(dim=-1)(logits.float()))).sum(dim=-1)[:,1:]-sum_ref_logits[:,1:]))* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # next_q = torch.cat([(((torch.log(torch.nn.Softmax(dim=-1)(old_logits.float())+1)).sum(dim=-1)[:,1:]-sum_ref_logits[:,1:]))* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # current_q = current_q + 0.9 * torch.cat([current_q[:,1:],torch.zeros_like(q_values[:,:1])],dim=1) + 0.8 * torch.cat([current_q[:,2:],torch.zeros_like(q_values[:,:2])],dim=1)
                # current_q = current_q / 3
                # current_q = sum_log_softmax(logits)#+torch.cat([torch.logsumexp(logits[:,:-1],dim=-1) - torch.logsumexp(logits[:,1:],dim=-1), torch.zeros_like(log_prob[:,:1])],dim=1)
                # next_q = torch.cat([current_q[:,1:],torch.zeros_like(current_q[:,:1])],dim=1)
                # current_q = sum_log_softmax(logits)-torch.logsumexp(torch.nn.Softmax(dim=-1)(logits),dim=-1)
                # sum_ref_logits = sum_log_softmax(ref_logits,q=True)
                # current_q = sum_exp(logits)
                # current_q = sum_mean(logits)
                # next_q = torch.cat([((current_q[:,1:]))* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # next_q = torch.cat([((current_q[:,1:]-torch.nn.LogSoftmax(dim=-1)(logits.float()).min()))* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                
                next_q = torch.cat([((current_q[:,1:]-old_logits[:,1:]))* response_mask[:,1:],torch.zeros_like(q_values[:,:1])],dim=1).detach()
                # q_values = torch.nn.LogSoftmax()(logits)-torch.nn.LogSoftmax()(ref_logits)
                
                # q_values = self.scale(logits.float())
                # q_values = (torch.exp(math.log(1000)*torch.nn.Softmax(dim=-1)(logits.float()))-1) / 1000
                # probs = torch.where(torch.nn.Softmax(dim=-1)(logits.float())>0.1,torch.nn.Softmax(dim=-1)(logits.float()),0)
                # q_values = torch.exp(math.log(2)*probs)-1
                # q_values = torch.exp(math.log(2)*torch.nn.Softmax(dim=-1)(logits.float()))#-torch.exp(math.log(2)*torch.nn.Softmax(dim=-1)(torch.ones_like(logits.float())))
                # q_values = torch.nn.LogSoftmax(dim=-1)(logits.float()).clip(0.)
                # print(torch.nn.Softmax(dim=-1)(logits.float()).max(),torch.nn.Softmax(dim=-1)(logits.float()).min(),torch.nn.Softmax(dim=-1)(logits.float()).mean())
                # q_values = torch.log(torch.nn.Softmax(dim=-1)(logits.float())*(math.e-1)+1) - 0.5 / logits.shape[-1]
                # q_values = torch.log(((logits.float()-logits.min()) / (logits.max()-logits.min()))+0.62)
                # q_values = torch.log(torch.nn.Sigmoid()(logits.float())*(math.e-1)+1)
                # q_values = self.scale(torch.log(torch.nn.Softmax(dim=-1)(logits.float())*(1e-1)+1),response_mask)##torch.log(torch.exp(torch.nn.LogSoftmax(dim=-1)(logits.float()))*(1e-1)+1)#torch.log(torch.nn.Softmax(dim=-1)(logits.float())*(1e-1)+1)#
                # q_values = (q_values - (0.095/q_values.shape[-1]))*10
                # q_values = (torch.nn.LogSoftmax(dim=-1)(logits.float())+1e).clip(0.)
                # q_values= torch.log(torch.nn.ReLU()(logits)+1).float()
                # non_zero_mask = (advantages != 0)
                # q_values = q_values * non_zero_mask.unsqueeze(-1)
                # torch.nn.Log(dim=-1)(logits.float()+1) # * 
                # ref_logits = self.ref_forward_micro_batch(ref_policy=ref,micro_batch=data, temperature=temperature)
                # logits = logits.float()
                # q_values = torch.log(((logits-logits.min())/(logits.max()-logits.min())*(1e1-1)+1+1e-6))# - torch.nn.LogSoftmax(dim=-1)(ref_logits).float()
                # q_values = torch.nn.Tanh()(logits.float()/(torch.abs(logits.float()).max(dim=-1,keepdim=True)[0]+1))
                # logits=logits.float()
                # self.actor_module.q_head_proj.requires_grad_(True) 
                # self.actor_module.print_info()
                # print(self.actor_module.q_head_proj.min(),self.actor_module.q_head_proj.mean(),self.actor_module.q_head_proj.max())
                # print(self.actor_module.q_head_proj.grad,self.actor_module.q_head_proj.is_leaf)
                # print(scale.grad,scale.is_leaf)
                # print(self.actor_module.q_head_proj,self.actor_module.q_head_proj.shape)
                # print(self.actor_module.lm_head.weight.shape)
                # q_values= torch.exp(torch.nn.LogSoftmax(dim=-1)(logits.float())) #torch.exp(torch.nn.LogSoftmax(dim=-1)(logits)).float()  * (scale.unsqueeze(0).unsqueeze(-1))
                # non_zero_mask = (advantages != 0)
                
                #q_values = q_values * response_mask.unsqueeze(-1)
                
                # q_values = q_values / (q_values.sum(dim=-1,keepdim=True) + 1)
                # c, _ = torch.max(logits, dim=-1, keepdim=True)
                # # print('-----',logits.max(),logits.min(),logits.mean(),'-----')
                # c = c.repeat(1,1,logits.shape[-1])
                # leaf_index = response_mask.sum(dim=-1,keepdim=True)-1#response_mask.size(1) - torch.argmax(response_mask.flip(1), dim=1) - 1
                # # print(logits.shape,leaf_index.shape)
                # leaf_logits = torch.gather(logits, 1, leaf_index.unsqueeze(-1).expand(-1, -1, logits.shape[-1])).repeat(1,logits.shape[1],1)
                # q_values = torch.exp(logits - c) / torch.sum(torch.exp(leaf_logits - c), dim=-1, keepdim=True) #torch.sum(torch.exp(logits - c), dim=1).reshape(-1, 1)
                # print(q_values.max(),q_values.min(),q_values.mean())
                # q_values = logits.float()
                # 
                # logits = logits.float()
                # scale = scale.to(logits.device, logits.dtype)
                # scale = torch.nn.ReLU()(scale)
                # leaf_index = response_mask.sum(dim=-1)-1
                # scale = scale.unsqueeze(0).unsqueeze(-1)
                # for i, k in enumerate(leaf_index):
                #     scale[i, k] = 1 
                # print(scale.shape)
                # q_values = logits - logits.logsumexp(dim=-1,keepdim=True)
                # q_values = torch.exp(torch.nn.LogSoftmax(dim=-1)(logits)) # * (scale.unsqueeze(0).unsqueeze(-1)) # * logits.max(dim=-1,keepdim=True)[0]
                # buckets = torch.linspace(-1, 1, q_values.shape[-1], device=q_values.device)
                # prob_ranks = torch.argsort(q_values,dim=-1)
                # result = torch.zeros_like(q_values)
                # bucket_idx = (prob_ranks / q_values.shape[-1] * (q_values.shape[-1] - 1)).long()
                # batch_indices = torch.arange(q_values.shape[0], device=q_values.device)[:, None, None].expand(q_values.shape[0], q_values.shape[1], q_values.shape[-1])
                # seq_indices = torch.arange(q_values.shape[1], device=q_values.device)[None, :, None].expand(q_values.shape[0], q_values.shape[1], q_values.shape[-1])
                # result[batch_indices, seq_indices, prob_ranks] = buckets[bucket_idx]
                # # result[prob_ranks] = buckets[bucket_idx]
                # q_values = result*q_values
                # q_values = q_values / (torch.exp(torch.std(q_values,dim=-1,keepdim=True))+1)
                # q_values = torch.nn.Tanh()(logits).float()
                # q_values = torch.nn.functional.normalize(logits,dim=-1).float()
                # q_values = ((q_values-q_values.min(dim=-1,keepdim=True)[0])/(q_values.max(dim=-1,keepdim=True)[0]-q_values.min(dim=-1,keepdim=True)[0]))*2-1
                # advantages = advantages.float()* torch.clamp(logits.float().max(),min=1)
                # q_values = (logits-logits.min(dim=-1,keepdim=True)[0])/(logits.max(dim=-1,keepdim=True)[0]-logits.min(dim=-1,keepdim=True)[0])
                # q_values = q_values.to(dtype=log_prob.dtype)
                # advantages = advantages.to(dtype=log_prob.dtype)
                # next_q = torch.cat([torch.sum(torch.topk(q_values[:,1:],k=20, dim=-1)[0],dim=-1),torch.zeros_like(q_values[:,:1,0])],dim=1)

                #next_q = torch.cat([q_values[:,1:].sum(dim=-1),torch.zeros_like(q_values[:,:1,0])],dim=1)
                
                # next_q = torch.cat([q_values[:,1:].max(dim=-1)[0],torch.zeros_like(q_values[:,:1,0])],dim=1)

                #q_values = torch.gather(q_values, -1, data['responses'].unsqueeze(-1)).squeeze(-1)
                
                # next_q = next_q / torch.std(logits,dim=-1)
                # print(q_values.shape, next_q.shape,advantages.shape)
                # next_q = torch.cat([q_values[:,1:],torch.zeros_like(q_values[:,:1])],dim=1)
                # advantages = torch.ones_like(advantages)
                # adv_mask = response_mask.clone()
                # adv_mask[response_mask.sum(dim=-1,keepdim=True)-1]=0
                # advantages = torch.where(adv_mask==1,0,advantages)
                #'''
                weights = torch.ones(advantages.shape,device=advantages.device, dtype=advantages.dtype)
                terminal_losses = []
                weights = torch.exp((current_q)).detach()*10
                '''
                for i in range(len(advantages)):
                    for j in range(len(advantages[i])-1):
                        
                        # if advantages[i,j].item()==100:
                        #     break
                        # if advantages[i,j].item()>0:
                        #     weights[i,j] = (1-q_values[i,j]) + 1
                        # if advantages[i].sum()>=0:
                        #     if next_q[i,j]>0:
                        #         weights[i,j] = (2-torch.exp(log_prob[i,j]))
                        #     else:
                        #         weights[i,j] = torch.exp(log_prob[i,j])-1
                        # else:
                        #     if next_q[i,j]>0:
                        #         weights[i,j] = torch.exp(log_prob[i,j])-1
                        if advantages[i,j+1]==0:
                            # terminal_losses.append((log_prob[i]* response_mask[i]).sum(dim=-1) * advantages[i,j].float())
                            
                            # response_mask.sum(dim=-1)[i]
                            # q_values[i,j] = ((log_prob)* response_mask*torch.exp((current_q.detach()))*10).sum(dim=-1)[i] / response_mask.sum(dim=-1)[i]#-(old_log_prob* response_mask).sum(dim=-1)[i]
                            # advantages[i,j] = q_values[i,j].detach() + advantages[i,j]
                            # torch.exp((current_q))*10
                            # if advantages[i,j]>0:
                            #     advantages[i,j] = 10.
                            # else:

                            # q_values[i,j] = q_values[i,j].detach()
                            # if advantages[i].sum()>0:
                            #     advantages[i,j]=-ref_log_prob[i,j]
                            # else:
                            #     advantages[i,j]=log_prob[i,j]
                            # elif advantages[i].sum()<0:
                            #     advantages[i,j]=advantages[i,j]+old_log_prob-ref_log_prob
                            # advantages[i,j] = torch.clamp(advantages[i,j].clone(),min=-0.1,max=0.1)
                            # if next_q[i,j]+advantages[i,j]>0:
                            #     weights[i,j] = (2-torch.exp(log_prob[i,j]))
                            # else:
                            #     weights[i,j] = torch.exp(log_prob[i,j])-1
                            # break
                            # if advantages[i,j].item()<0:
                            #     advantages[i,j]=max(advantages[i,j],-0.001)#*(1-min(1,self.step/100))
                            #     if self.step < 20:
                            #         advantages[i,j] = advantages[i,j] / 2
                            # else:
                            #     advantages[i,j]=advantages[i,j]*(1-min(1,self.step/100))+0.4
                            
                            # elif advantages[i,j].item()==0:
                            #     advantages[i,j]=0.25*(1-min(1,self.step/300))-0.31
                            # elif advantages[i,j].item()==0.1:
                            #     advantages[i,j]=0.15*(1-min(1,self.step/300))-0.17
                            break
                        # if advantages[i,j]==1:
                        #     advantages[i,j] == 1*(1-min(1,self.step/300))
                        # else:
                        advantages[i,j] = 0 #TODO
                        # q_values[i,j] = 0.
                        # if next_q[i,j]+advantages[i,j]>0:
                        #     weights[i,j] = (2-torch.exp(log_prob[i,j]))
                        # else:
                        #     weights[i,j] = torch.exp(log_prob[i,j])-1
                        # q_values[i,j] = q_values[i,j]/ ((logits.float()/100).sum(dim=-1)-sum_ref_logits)[i,j]
                        # elif advantages[i,j]==0.1:
                        #     advantages[i,j] == 0.5
                        # elif advantages[i,j]==0:
                        #     advantages[i,j] == -0.5
                        # advantages[i,j]=advantages[i,j]*(1-min(1,self.step/300))
                        # if advantages[i,j]==10:
                        #     advantages[i,j] == 10*(1-min(1,self.step/300))
                        # else:
                        #     advantages[i,j]=0.5*advantages[i,j]*(1-min(1,self.step/300)) #q_values[i,j].clone().detach()-next_q[i,j].clone().detach()#
                        
                       
                    '''
                target = advantages.float() + next_q
                # target = q_values.detach() + target
                # target = torch.clamp(target, min=-1.5,max=1.5)
                # print(">>>>>",advantages.max())
                # print(">>>>>",response_mask.sum(dim=-1),torch.gather(q_values.float(), 1, response_mask.sum(dim=-1,keepdim=True)-1),torch.gather(q_values.float(), 1, response_mask.sum(dim=-1,keepdim=True)))
                rpe_loss = core_algos.compute_rpe_loss(q_values.float(), target.float(), advantages, eos_mask=response_mask,step=self.step, weights=weights)
                # core_algos.compute_rpe_loss(q_values.float(), target.float(), eos_mask=response_mask,step=self.step, weights=weights)- sum(terminal_losses)/len(terminal_losses)
                # rpe_loss = rpe_loss + terminal_loss
                 # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                #                                                               log_prob=log_prob,
                #                                                               advantages=advantages,
                #                                                               eos_mask=response_mask,
                #                                                               cliprange=clip_ratio)
                # # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                # rpe_loss = rpe_loss - entropy_loss * entropy_coeff# * 10# * (0.02+(1-min(1,self.step/300)))
                # # compute policy loss
                # policy_loss = pg_loss - entropy_loss * entropy_coeff * 50
                '''
                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # ref_log_prob = logprobs_from_logits(data['ref_logits'], data['responses'])
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)
                    rpe_loss = rpe_loss + kl_loss * self.config.kl_loss_coef
                    # policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef
                '''
                # loss = policy_loss / self.gradient_accumulation
                loss = rpe_loss / self.gradient_accumulation
                loss.backward()
               
                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/target_mean':verl_F.masked_mean(target, response_mask).detach().item(),
                    'actor/target_min':target.min().detach().item(),
                    'actor/target_max':target.max().detach().item(),
                    # 'actor/sum_mean':sum_ref_logits[:,1:].mean().detach().item(),
                    # 'actor/sum_max':sum_ref_logits[:,1:].max().detach().item(),
                    # 'actor/sum_min':sum_ref_logits[:,1:].min().detach().item(),
                    # 'actor/pg_loss': pg_loss.detach().item(),
                    # 'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    # 'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/rpe_loss': rpe_loss.detach().item(),
                    # 'actor_q/q_value': q_values[0].tolist(),
                    # 'actor_q/next_q_value': next_q[0].tolist(),
                    
                }
                append_to_dict(metrics, data)
           
            grad_norm = self._optimizer_step()
            # print(">>>>>", self.scale.para[0][0])
            data = {'actor/grad_norm': grad_norm.detach().item(),
            }
        
            append_to_dict(metrics, data)
        # data_q = [[x, y] for (x, y) in zip(torch.arange(len(next_q[0])).cpu(), next_q[0].detach().cpu())]
        # table_q_value = wandb.Table(data=data_q, columns=["position", "q_value"])
        # data_next_q = [[x, y] for (x, y) in zip(torch.arange(len(next_q[0])).cpu(), next_q[0].detach().cpu())]
        # table_next_q = wandb.Table(data=data_next_q, columns=["position", "q_value"])
        # data = {'actor_q/q_value': table_q_value}
        #     # 'actor_q/next_q_value': table_next_q,}
        # append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        self.step+=1
        # self.scale_optimizer.zero_grad()
        return metrics