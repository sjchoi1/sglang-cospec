#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/nsys_profiles"
mkdir -p "$OUTPUT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: profile.sh <command> [args...]"
    echo "Example: profile.sh bash server.sh"
    exit 1
fi

# Use first arg basename (without extension) as profile name
PROFILE_NAME="$(basename "${1}" .sh)"

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --sample=cpu \
    --cuda-memory-usage=true \
    --output="$OUTPUT_DIR/$PROFILE_NAME" \
    --force-overwrite=true \
    "$@"
