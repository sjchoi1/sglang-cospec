#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============ Configuration ============
URL="http://localhost:30000"
NUM_PROMPTS=100
REQUEST_RATE="inf"
PROFILE=false
PROFILE_NUM_STEPS=5
PROFILE_OUTPUT_DIR="./traces"
# =======================================

if [ "$PROFILE" = true ]; then
    python -m sglang.profiler \
        --url "$URL" \
        --num-steps "$PROFILE_NUM_STEPS" \
        --output-dir "$PROFILE_OUTPUT_DIR" \
        --cpu --gpu &
    PROFILE_PID=$!
fi

python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$URL" \
    --dataset-name sharegpt \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    2>&1 | tee "$SCRIPT_DIR/client.log"

if [ "$PROFILE" = true ]; then
    wait "$PROFILE_PID"
    echo "Profiling complete. Check $PROFILE_OUTPUT_DIR for traces."
fi
