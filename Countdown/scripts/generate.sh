export N_GPUS=4
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=countdown-rover-3b-mse-in2-e1
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.98
# STEPS=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400)
STEPS=(100)
BASE_OUTPUT_DIR=""
MODEL_BASE_PATH=""
TEST_DATA_PATH=""
for i in "${STEPS[@]}"; do
    # Create output directory if not exists
    OUTPUT_DIR="${BASE_OUTPUT_DIR}-${i}"
    MODEL_PATH="${MODEL_BASE_PATH}"
    N_PASSES=64
    TP_SIZE=1
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=${TEST_DATA_PATH} \
        data.output_path=${OUTPUT_DIR}/result.parquet \
        data.n_samples=${N_PASSES} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=1.0 \
        rollout.response_length=1024 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.8 \
        rollout.tensor_model_parallel_size=${TP_SIZE}
done
