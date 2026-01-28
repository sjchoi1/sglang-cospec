#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-cospec}"
IMAGE="nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"

# Commands to run inside docker
read -r -d '' SETUP_COMMANDS << 'EOF' || true
set -e
apt-get update
apt-get install -y python3 python3-pip git
pip3 install --upgrade pip
cd /workspace/sglang/python
pip3 install -e ".[all]"
cd /workspace/sglang
exec bash
EOF

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Reusing existing container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" 2>/dev/null || true
    docker exec -it -w /workspace/sglang "$CONTAINER_NAME" bash
else
    echo "Creating new container: $CONTAINER_NAME"
    docker run -it \
        --gpus all \
        --name "$CONTAINER_NAME" \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$SGLANG_ROOT":/workspace/sglang \
        -w /workspace/sglang \
        "$IMAGE" \
        bash -c "$SETUP_COMMANDS"
fi
