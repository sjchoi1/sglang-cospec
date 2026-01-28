# CoSpec: Colocated Speculative Decoding

Pipelined draft/target execution with SM partitioning for speculative decoding.

## Overview

CoSpec pipelines draft and target model execution on the same GPU:

```
Time T:   draft(batch_N)   ||  verify(batch_N-1)
Time T+1: draft(batch_N+1) ||  verify(batch_N)

draft_stream  (SMs 0-20%):   [Draft N] [Draft N+1] [Draft N+2] ...
target_stream (SMs 20-100%):      [Verify N-1] [Verify N] [Verify N+1] ...
```

## Files

### Core Implementation (in `python/sglang/srt/`)

| File | Description |
|------|-------------|
| `speculative/colocated_standalone_worker.py` | Main worker with pipelined execution |
| `speculative/spec_info.py` | Algorithm enum (COLOCATED added) |
| `utils/sm_controller.py` | SM partitioning via libsmctrl |

### This Directory

```
cospec_scripts/
├── README.md              # This file
├── build_libsmctrl.sh     # Build SM partitioning library
├── bench_spec.py          # Benchmarking script
├── run_docker.sh          # Docker helper
└── libsmctrl/
    ├── src/
    │   ├── libsmctrl.c    # SM control library source
    │   └── libsmctrl.h
    └── build/
        └── libsmctrl.so   # Built library (after running build script)
```

## Setup

### 1. Build libsmctrl.so

```bash
cd cospec_scripts
bash build_libsmctrl.sh
```

### 2. Run with COLOCATED algorithm

```bash
python -m sglang.launch_server \
    --model-path <target_model> \
    --speculative-algorithm COLOCATED \
    --speculative-draft-model-path <draft_model> \
    --speculative-num-steps 5
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSPEC_DRAFT_SM_RATIO` | `0.2` | Fraction of SMs for draft model (0.0-1.0) |

Example:
```bash
export COSPEC_DRAFT_SM_RATIO=0.3  # 30% draft, 70% target
```

## How It Works

### SM Partitioning

Uses `libsmctrl` to assign different SMs to different CUDA streams:

```python
# draft_stream uses TPCs 0-13 (SMs 0-26)
sm_ctrl.set_stream_mask(draft_stream, 0, 13)

# target_stream uses TPCs 13-66 (SMs 26-132)
sm_ctrl.set_stream_mask(target_stream, 13, 66)
```

This enables true concurrent execution within a single process.

### Pipeline Flow

```
forward_batch_generation(batch_N):
    1. If pending_batch exists:
       - Launch verify(pending_batch) on target_stream
    2. Launch draft(batch_N) on draft_stream
    3. Synchronize both streams
    4. Store batch_N as pending for next iteration
    5. Return verify result
```

## Dependencies

- CUDA toolkit (for building libsmctrl)
- PyTorch with CUDA support
- SGLang
