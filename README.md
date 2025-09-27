### Start ROVER Training

```
./scripts/deepscaler/train/rover_qwen3_8b.sh --model /path/to/qwen3-8b --wandb_api_key your_wandb_api_key
```

### Multi-node training

To accelerate the training process, we recommend using at least 4 nodes. Our multi-node training is based on Ray. You can run `ray start --head` on the head node and `ray start --address=[RAY_ADDRESS]` on other nodes to start the Ray cluster.

After starting the clusterrun the training script on the head node.

