#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=your_wandb_api_key
export WANDB_MODE=offline

N_NODE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --n_node)
            N_NODE="$2"
            shift 2
            ;;
        --wandb_api_key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

export MODEL_PATH=$MODEL_PATH  # for the correct reward function in math_reward.py

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH must be specified!"
    exit 1
fi

SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME_NO_EXT="${SCRIPT_NAME%.*}"
CKPTS_DIR=${CKPTS_DIR:-"ckpts/${SCRIPT_NAME_NO_EXT}"}

# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m verl.trainer.main_rpe \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=/data/ROVER-qwen3/scripts/deepscaler/data/deepscaler_train.parquet \
    data.val_files=/data/ROVER-qwen3/scripts/deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.truncation="left" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-0827' \
    trainer.experiment_name="${SCRIPT_NAME_NO_EXT}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$N_NODE \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=30 "${@:1}"
