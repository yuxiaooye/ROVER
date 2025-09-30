
DEBUG=False
if [ $# -eq 0 ]; then
    echo "Error: Please provide GLOBAL_STEP_DIR"
    echo "Usage: $0 <GLOBAL_STEP_DIR>"
    exit 1
fi

if [ $# -eq 1 ]; then
    echo "Error: Please provide EVAL_TYPE (e.g., AIME2024)"
    echo "Usage: $0 <GLOBAL_STEP_DIR> <EVAL_TYPE>"
    exit 1
fi

GLOBAL_STEP_DIR=$1
EVAL_TYPE=$2
BSZ=${3:-96}
TEMPERATURE=${4:-0.6}
VERSION=${5:-"qwen3"}

MODEL_NAME=$(basename $(dirname $GLOBAL_STEP_DIR) | tr '_' '-' | tr '.' 'p' | tr '[:upper:]' '[:lower:]')  

if [ "$DEBUG" = "True" ]; then
    TEST_FILE=benchmarks/aime-2024-deepscaler.parquet
else
    if [ "$EVAL_TYPE" = "AIME2024" ]; then
        TEST_FILE=benchmarks/aime-2024-deepscaler-repeat256.parquet
    elif [ "$EVAL_TYPE" = "AIME2025" ]; then
        TEST_FILE=benchmarks/aime-2025-deepscaler-repeat256.parquet
    elif [ "$EVAL_TYPE" = "HMMT2025" ]; then
        TEST_FILE=benchmarks/hmmt-2025-deepscaler-repeat256.parquet
    elif [ "$EVAL_TYPE" = "OLYMPIAD" ]; then
        TEST_FILE=benchmarks/olympiad_bench-deepscaler-repeat15.parquet
    elif [ "$EVAL_TYPE" = "AMC2023" ]; then
        TEST_FILE=benchmarks/amc-2023-deepscaler-repeat64.parquet
    elif [ "$EVAL_TYPE" = "MATH500" ]; then
        TEST_FILE=benchmarks/math500-deepscaler-repeat5.parquet
    else
        echo "Error: Please check EVAL_TYPE"
        echo "Current value: $EVAL_TYPE"
        exit 1
    fi
fi

# initialize conda
source /data/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate rover-math

# # model convert to hf
cd ..
sudo python -m verl.model_merger.fsdp_model_merger --global_step_dir $GLOBAL_STEP_DIR
sudo rm -rf $GLOBAL_STEP_DIR/actor_merged/generation_config.json

# vllm deploy
vllm serve \
    $GLOBAL_STEP_DIR/actor_merged \
    --served-model-name $MODEL_NAME \
    --port 8080 \
    --tensor-parallel-size 8 \
    --dtype auto \
    --api-key token-abc123 \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching > /dev/null 2>&1 &

# wait for vllm to be ready
echo " waiting vllm to be ready..."
MAX_RETRIES=60
RETRY_COUNT=0
RETRY_INTERVAL=10
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # check vllm health status
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer token-abc123" http://localhost:8080/v1/models 2>/dev/null)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ vllm is ready, start eval..."
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo " waiting vllm to be ready... (try $RETRY_COUNT/$MAX_RETRIES)"
            sleep $RETRY_INTERVAL
        fi
    fi
done
if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo " error: vllm is not ready in $((MAX_RETRIES * RETRY_INTERVAL)) seconds"
    exit 1
fi

# rollout
conda activate rover-math-eval
cd eval
if [ "$DEBUG" = "True" ]; then
    python gen_vllm.py --model $MODEL_NAME --test_file $TEST_FILE --max_tokens 20480 --batch_size $BSZ --temperature $TEMPERATURE
else
    python gen_vllm.py --model $MODEL_NAME --test_file $TEST_FILE --max_tokens 20480 --batch_size $BSZ --temperature $TEMPERATURE
fi

# eval
cd eval
if [ "$EVAL_TYPE" = "OLYMPIAD" ]; then
    python eval_math_verify.py --model $MODEL_NAME --test_file $TEST_FILE --temperature $TEMPERATURE --is_olympiad
else
    python eval_math_verify.py --model $MODEL_NAME --test_file $TEST_FILE --temperature $TEMPERATURE
fi