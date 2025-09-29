export N_GPUS=4
export BASE_MODEL=<Your model path>
export DATA_DIR=./data/countdown
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-rover
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.98
export WANDB_MODE=offline
bash scripts/train_rover.sh
