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

## Adaptive Colocated Speculation (Planned)

### The Fundamental Trade-off

The effectiveness of colocated speculative decoding depends on **compute-boundedness (β)**, which is determined by batch size (B) and sequence length (L):

```
β (compute-boundedness) = f(B, L)

- Large B, short L → HIGH β (compute-bound)
- Small B, long L  → LOW β (memory-bound)
```

**The core tension:**

```
                    MEMORY-BOUND ←─────────────────→ COMPUTE-BOUND
                    (low β)                          (high β)
                         │                              │
    Speculation value:   HIGH ────────────────────→ LOW
    (γ benefit)          (fills memory stalls)     (adds compute overhead)
                         │                              │
    Colocation overhead: HIGH ────────────────────→ LOW
    (relative impact)    (visible vs fast compute) (hidden by compute)
```

**Key insight:** The same factor (β) determines BOTH:
1. How much speculation (γ) is beneficial
2. Whether colocation overhead is acceptable

### Compute-Boundedness Model

```
β(B, L) = T_compute(B, L) / T_memory(B, L)

Where:
  T_compute ∝ B × d²           # GEMM scales with batch × hidden²
  T_memory  ∝ B × L × d        # Attention KV access scales with seq_len

Simplified for fixed model:
  β ∝ B / L

  - High B/L ratio → compute-bound → low γ, colocation overhead hidden
  - Low B/L ratio  → memory-bound  → high γ, but colocation overhead visible
```

### Operating Regimes

```
        Sequence Length (L)
        LONG │
             │  ┌─────────────────────────────────────┐
             │  │ MEMORY-BOUND (low β)                │
             │  │ • Speculation valuable (high γ)     │
             │  │ • Colocation overhead significant   │
             │  │ → High γ, but maybe sequential      │
             │  └──────────────────┬──────────────────┘
             │                     │
             │  ┌──────────────────▼──────────────────┐
             │  │ SWEET SPOT (moderate β)             │
             │  │ • Speculation still valuable        │
             │  │ • Colocation overhead acceptable    │
             │  │ → Colocated speculation, tune (r,γ) │
             │  └──────────────────┬──────────────────┘
             │                     │
             │  ┌──────────────────▼──────────────────┐
             │  │ COMPUTE-BOUND (high β)              │
             │  │ • Speculation overhead > benefit    │
             │  │ • Colocation overhead hidden        │
             │  │ → Reduce γ toward 0, no speculation │
       SHORT │  └─────────────────────────────────────┘
             └────────────────────────────────────────────→
                SMALL                              LARGE
                              Batch Size (B)
```

### Two-Level Optimization

**Level 1: Mode Selection (based on β)**
- Compute β from (B, L)
- Determine γ (decreases with β)
- Determine whether colocation is beneficial

**Level 2: Timing Match (within colocated mode)**
- Given (B, L, γ), find SM ratio r such that T_draft ≈ T_target
- This maximizes GPU utilization by eliminating idle time

```
┌─────────────────────────────────────────────────────────────┐
│  Input: (B, L)                                              │
│     │                                                       │
│     ▼                                                       │
│  β = compute_boundedness(B, L)                              │
│     │                                                       │
│     ├──────────────────────────────────────────┐            │
│     ▼                                          ▼            │
│  γ* = f(β)                              coloc* = g(β)       │
│  (speculation depth)                    (use colocation?)   │
│     │                                          │            │
│     └──────────────┬───────────────────────────┘            │
│                    ▼                                        │
│              If coloc* = True:                              │
│                 r* = argmin |T_draft(r,γ*) - T_target(r,γ*)|│
│                 (SM ratio for timing match)                 │
└─────────────────────────────────────────────────────────────┘
```

### Objective Function

```
Maximize:  Throughput = (B × γ × α) / T_iteration

Where:
  T_iteration = max(T_draft, T_target)           if colocated
              = T_draft + T_target               if sequential
              = T_decode                         if no speculation

  T_draft(r, γ)  = γ × f_draft(r × S, B, L)
  T_target(r, γ) = f_target((1-r) × S, B × γ, L)

  α = acceptance rate
  S = total SMs

Optimal configuration (γ*, coloc*, r*) maximizes throughput.
```

### Adaptive Algorithm

```python
class AdaptiveCoSpecController:
    """
    Unified controller for speculation depth, colocation, and SM ratio.
    All decisions driven by compute-boundedness β = f(B, L).
    """

    def __init__(self, model_config):
        self.d = model_config.hidden_dim
        self.γ_max = 8
        self.k_gamma = 0.5      # γ decay rate with β
        self.β_sweet_low = 0.3  # Colocation sweet spot bounds
        self.β_sweet_high = 2.0

    def compute_beta(self, B, L):
        """Compute-boundedness from batch size and sequence length."""
        # Attention is memory-bound (scales with L)
        # FFN is compute-bound (scales with B)
        return (B * self.d) / (L * MEM_BANDWIDTH)

    def decide(self, B, L, acceptance_rate):
        """
        Determine (γ, use_colocation, r) based on (B, L).
        """
        β = self.compute_beta(B, L)

        # Step 1: Determine γ (decreases with compute-boundedness)
        γ = int(self.γ_max * math.exp(-self.k_gamma * β))
        γ = max(0, min(self.γ_max, γ))

        # Adjust for low acceptance rate
        if acceptance_rate < 0.5:
            γ = max(0, γ - 2)

        # Step 2: Determine colocation (sweet spot in moderate β)
        if γ == 0:
            use_colocation = False
            r = None
        elif β < self.β_sweet_low:
            # Memory-bound: speculation good, but colocation overhead high
            # Use sequential speculation
            use_colocation = False
            r = None
        elif β > self.β_sweet_high:
            # Compute-bound: low γ anyway, colocation overhead hidden
            # Could colocate but little benefit
            use_colocation = (γ >= 2)
            r = 0.1 if use_colocation else None
        else:
            # Sweet spot: both speculation and colocation beneficial
            use_colocation = True
            r = self.find_optimal_r(B, L, γ)

        return CoSpecConfig(gamma=γ, use_colocation=use_colocation, sm_ratio=r)

    def find_optimal_r(self, B, L, γ):
        """
        Find SM ratio r such that T_draft ≈ T_target.
        Uses online measurement and gradient adjustment.
        """
        # Start with heuristic, refine with measurements
        r = 0.2

        # Online adjustment based on timing measurements
        if hasattr(self, 'ema_t_draft') and hasattr(self, 'ema_t_target'):
            imbalance = (self.ema_t_draft - self.ema_t_target) / \
                        max(self.ema_t_draft, self.ema_t_target)
            if imbalance > 0.1:    # Draft slower
                r += 0.02
            elif imbalance < -0.1: # Target slower
                r -= 0.02
            r = max(0.05, min(0.4, r))

        return r

    def update_measurements(self, t_draft, t_target):
        """Update EMA of timing measurements for r optimization."""
        α = 0.2
        self.ema_t_draft = α * t_draft + (1-α) * getattr(self, 'ema_t_draft', t_draft)
        self.ema_t_target = α * t_target + (1-α) * getattr(self, 'ema_t_target', t_target)
```

### Multi-GPU Extension

With tensor parallelism (TP), additional factors affect β:

```
Multi-GPU (TP=N):
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │
│ D │ T   │ │ D │ T   │ │ D │ T   │ │ D │ T   │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     └───────────┴─ AllReduce ┴───────────┘
```

**Modified compute-boundedness:**

```
β_multi(B, L, TP) = β_single(B, L) / TP + comm_overhead(TP)

Where:
  comm_overhead(TP) = AllReduce_time / compute_time

High TP → lower per-GPU compute → more memory-bound per GPU
         BUT communication overhead increases
```

**Draft model placement strategies:**

| Strategy | When to use | β effect |
|----------|-------------|----------|
| Sharded (like target) | Large draft model | Adds draft AllReduce |
| Replicated per GPU | Small draft model | No draft AllReduce, memory cost |
| Subset of GPUs | Heterogeneous | Reduces draft comm |

### Why Coarse-Grained (Not Kernel-Level)

We adjust per-iteration, not per-kernel:

1. **Temporal misalignment**: Draft and target kernels don't execute in lockstep
2. **Overhead**: Per-kernel SM mask changes (~2μs × thousands) exceeds gains
3. **Macro-balance suffices**: T_draft ≈ T_target over iteration is sufficient
4. **Hardware scheduling**: GPU optimizes within each partition

### Expected Benefits

| β Regime | Static Config | Adaptive Config |
|----------|---------------|-----------------|
| Low (memory-bound) | Coloc overhead hurts | Sequential spec, high γ |
| Moderate (sweet spot) | Suboptimal (r, γ) | Optimal colocated spec |
| High (compute-bound) | Wasted speculation | Auto-reduce γ → 0 |
| Variable workload | Always suboptimal | Adapts to current (B, L) |

Estimated improvement: **15-30% throughput gain** from regime-aware optimization.

### References

- [BulletServe](https://arxiv.org/abs/2504.19516): Dynamic SM partitioning for prefill/decode
- [libsmctrl](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/): SM masking library
- [SpecInfer](https://arxiv.org/abs/2305.09781): Tree-based speculative inference
- [EAGLE](https://arxiv.org/abs/2401.15077): Feature-aligned draft models

## Dependencies

- CUDA toolkit (for building libsmctrl)
- PyTorch with CUDA support
- SGLang
