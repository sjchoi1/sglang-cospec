#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============ Configuration ============
MODEL_PATH="Qwen/Qwen3-8B"
DRAFT_MODEL_PATH="Qwen/Qwen3-0.6B"
SPEC_ALGORITHM="COLOCATED"
# SPEC_ALGORITHM="STANDALONE"
SPEC_NUM_STEPS=3
SPEC_EAGLE_TOPK=1
SPEC_NUM_DRAFT_TOKENS=4
MEM_FRACTION_STATIC=0.50
PORT=30000
# =======================================

COSPEC_DRAFT_SM_RATIO=0.3 COLOCATED_MIN_BATCH_SIZE=1 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --speculative-algorithm "$SPEC_ALGORITHM" \
    --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
    --speculative-num-steps "$SPEC_NUM_STEPS" \
    --speculative-eagle-topk "$SPEC_EAGLE_TOPK" \
    --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --port "$PORT" \
    2>&1 | tee "$SCRIPT_DIR/server.log"
