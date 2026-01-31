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

## Analytical Throughput Model for Adaptive Colocated Speculation

### 1. Problem Statement

Colocated speculative decoding jointly optimizes three coupled outputs given three runtime inputs:

```
Inputs (observed at runtime, not controllable):
  B = batch size
  L = sequence length
  α = acceptance rate (fraction of draft tokens accepted by target)

Outputs (jointly coupled):
  γ* = speculation depth (number of draft steps per iteration)
  c* = colocation mode  (colocated vs. sequential vs. AR-only)
  r* = SM ratio          (fraction of SMs for draft, only when c*=colocated)
```

**Key regimes**:
- **High B hurts speculation**: at high batch sizes, speculative decoding is *worse* than
  autoregressive (AR) decoding — the draft model consumes compute that yields diminishing
  acceptance benefit while the target verification cost grows linearly with γ. An adaptive
  system must recognize this regime and fall back to AR.
- **Long L helps speculation**: at long sequence lengths, both draft and target become
  memory-bound (dominated by KV cache reads). In this regime the draft model's marginal
  compute cost is near-zero — the memory bus is the bottleneck regardless — so the extra
  draft steps come "for free" while still producing E(γ, α) > 1 expected tokens per
  iteration. Long L therefore expands the region where speculation is beneficial.

**Batch splitting overhead and colocation**:

Colocation pipelines draft and target on *different sub-batches*, which requires splitting
the batch. This introduces overhead — but the overhead is **layer-type dependent**:

```
FFN layers (compute-bound):
  Cost = 2 × W_read + compute(B/2)    (split)
  Cost = 1 × W_read + compute(B)      (unsplit)

  Overhead = extra weight read (W_read).
  At high B, compute(B) >> W_read, so the extra read is amortized away.
  → High B reduces FFN splitting overhead.

Attention layers (memory-bound):
  Cost ∝ KV_read(B/2, L) + KV_read(B/2, L)  (split)
  Cost ∝ KV_read(B, L)                        (unsplit)

  KV cache is per-sequence — splitting the batch does NOT re-read the same KV entries.
  Each half reads its own KV, same total bytes either way.
  → Attention splitting has near-zero overhead regardless of B.
```

This yields a key insight connecting speculation and colocation:

```
The traditional view: spec decoding is a compute-vs-memory tradeoff.
  - Compute-bound (high B) → spec loses to AR (draft overhead > acceptance gain)
  - Memory-bound (low B, long L) → spec wins (draft is "free")

The colocation addition: batch splitting overhead follows the SAME axis.
  - Compute-bound (high B) → splitting overhead is small (weight reads amortized)
  - Memory-bound (long L) → splitting overhead is small (attention dominates, no penalty)

Both conditions that make speculation beneficial ALSO make colocation cheap:
  - Long L → memory-bound → spec wins AND attention dominates → splitting is free
  - High B → compute-bound → spec loses BUT splitting overhead is also low
                              (so if spec is still marginal, colocation costs little to try)
```

This means the (B, L) regions where colocation is *most* useful (speculation is beneficial)
are precisely the regions where its overhead (batch splitting) is *lowest* — a fortunate
alignment that makes colocated speculation robust across the operating space.

### 2. Throughput Model

We model draft and target step latency using a **roofline-style** formulation.
Each step is either memory-bound or compute-bound depending on the arithmetic
intensity relative to the hardware's compute-to-bandwidth ratio.

```
Hardware parameters:
  S     = total SMs on the GPU
  F     = peak FLOP/s (per SM)
  M     = memory bandwidth (bytes/s)
  β_hw  = S × F / M                          # machine balance point (FLOP/byte)

Per-step latency (roofline):
  t_step(s, B, L, P) = max(
      P × B / (s × F),                       # compute-bound term
      P × sizeof(param) / M + B × L × sizeof(kv) / M   # memory-bound term
  )

  where:
    s = number of SMs allocated to this model
    P = model parameter count
    B = batch size
    L = sequence length (context for KV cache)

Draft step:   t_d(s, B, L) = t_step(s, B, L, P_draft)
Target step:  t_t(s, B, L) = t_step(s, B, L + γ, P_target)
```

**Throughput for each execution mode:**

```
Mode 1 — AR (no speculation, γ = 0):
  T_AR = B / t_t(S, B, L)

Mode 2 — Sequential speculative decoding:
  T_seq(γ) = B × E(γ, α) / [γ × t_d(S, B, L) + t_t(S, B, L + γ)]

Mode 3 — Colocated speculative decoding:
  T_col(γ, r) = B × E(γ, α) / max(γ × t_d(rS, B, L),  t_t((1-r)S, B, L + γ))

Expected accepted tokens (geometric acceptance model):
  E(γ, α) = (1 - α^(γ+1)) / (1 - α)         when α < 1
           = γ + 1                             when α = 1
```

### 3. Phase Boundary Analysis

The input space (B, L, α) is partitioned into regions where different modes are optimal.
The boundaries between these regions can be derived analytically.

**Boundary 1: B_crit(L, α) — AR overtakes speculation**

Speculation is beneficial only when extra accepted tokens outweigh the iteration cost:

```
T_seq(γ) > T_AR  requires:

  E(γ, α) / [γ × t_d(S, B, L) + t_t(S, B, L + γ)]  >  1 / t_t(S, B, L)

Rearranging:
  E(γ, α) × t_t(S, B, L)  >  γ × t_d(S, B, L) + t_t(S, B, L + γ)
```

At large B, both models become compute-bound. The draft overhead `γ × t_d` grows
linearly with γ, while `E(γ, α)` saturates (bounded by `1/(1-α)`). There exists a
critical batch size `B_crit` above which the inequality fails for all γ > 0:

```
B_crit(L, α) ≈ M / (P_draft × F) × [E(γ*, α) - 1] × t_t(S, B, L) / t_d(S, B, L)
```

In practice, `B_crit` decreases as α decreases (poor acceptance makes speculation
less worthwhile at smaller batch sizes).

**Boundary 2: β_crit(γ) — colocation beats sequential**

Colocation is beneficial when overlapping draft and target hides latency:

```
T_col(γ, r*) > T_seq(γ)  requires:

  max(γ × t_d(rS, ...), t_t((1-r)S, ...))  <  γ × t_d(S, ...) + t_t(S, ...)
```

Define the compute-boundedness ratio β = (compute time) / (memory time) for the
target model. When β > 1 (compute-bound), reducing SMs via `(1-r)` increases
target latency proportionally. When β < 1 (memory-bound), reducing SMs has
minimal effect on target latency — making colocation nearly free.

```
β_crit(γ):  colocation wins when β_target < β_crit, i.e., when the target is
            sufficiently memory-bound that SM reduction doesn't hurt it much.

β_crit ≈ γ × t_d(S) / t_t(S)    (first-order approximation)
```

**The (B, L) phase plane:**

```
          L (sequence length)
          │
          │   ┌─────────────────────┐
          │   │                     │
          │   │   COLOCATED SPEC    │
          │   │   (memory-bound     │
          │   │    regime)          │
          │   │                     │
          │   ├─────────────────────┤ ← β = β_crit boundary
          │   │  SEQUENTIAL SPEC    │
          │   │  (compute-bound     │
          │   │   regime)           │
          │   ├─────────────────────┤ ← B = B_crit(L, α) boundary
          │   │                     │
          │   │   AR ONLY           │
          │   │   (high batch,      │
          │   │    speculation      │
          │   │    not worthwhile)  │
          │   │                     │
          │   └─────────────────────┘
          └──────────────────────────→ B (batch size)

As α decreases, B_crit shifts left (AR region expands).
As α increases, B_crit shifts right (speculation stays beneficial longer).
```

### 4. Structural Properties

Two structural properties of the throughput surface enable efficient online optimization
without exhaustive search:

**Property 1: Unimodality of throughput in r (for fixed γ, B, L)**

For fixed γ in colocated mode, throughput `T_col(γ, r)` is unimodal in r.

```
Proof sketch:
  T_col ∝ 1 / max(γ × t_d(rS), t_t((1-r)S))

  - t_d(rS) is monotone decreasing in r (more SMs → faster draft)
  - t_t((1-r)S) is monotone increasing in r (fewer SMs → slower target)
  - Their max has a unique minimum at the crossing point r* where:
      γ × t_d(r*S) = t_t((1-r*)S)
  - Away from r*, one term dominates and throughput decreases.
```

This means binary search or gradient ascent on r converges to the optimum.

**Property 2: Diminishing marginal returns of γ**

The marginal throughput gain of increasing γ by 1 is:

```
  ΔE(γ) = E(γ+1, α) - E(γ, α) = α^(γ+1)

  This decays exponentially in γ.
```

Meanwhile, the marginal cost (extra draft step) is roughly constant. Therefore
the net marginal benefit `ΔE(γ) / Δcost` is monotone decreasing, and γ* can be
found by stopping when the marginal gain falls below the marginal cost — no need
to evaluate all γ values.

### 5. Online Controller

The analytical boundaries and structural properties enable a lightweight controller
that adapts at runtime without offline profiling:

```
OnlineController(B, L, α, timing_feedback):

  1. MODE SELECTION (analytical boundaries):
     if B > B_crit(L, α):
         return AR mode (γ = 0)
     if β_target < β_crit(γ_est):
         mode = COLOCATED
     else:
         mode = SEQUENTIAL

  2. SPECULATION DEPTH (marginal analysis):
     γ* = max γ such that α^γ > cost_threshold
        = floor(-log(cost_threshold) / -log(α))
     (closed-form from diminishing returns property)

  3. SM RATIO TUNING (gradient-based, colocated only):
     Use timing feedback (T_draft_observed, T_target_observed)
     to do gradient ascent on r:
       imbalance = T_draft - T_target
       r ← r + η × sign(imbalance)
     Converges due to unimodality (Property 1).

  Runtime cost: O(1) per iteration (no search, no profiling).
```

**Adaptation to workload shifts:**

The controller tracks EMA statistics of B, L, α and recomputes boundaries when
the operating point crosses a phase boundary. Mode switches are hysteresis-gated
to avoid oscillation.

### 6. Evaluation Plan

**Baselines:**
- Static SpecInfer (fixed γ, sequential)
- Static Eagle (fixed γ, sequential)
- Fixed CoSpec (fixed γ, fixed r, colocated)
- Oracle (exhaustive offline search per workload point)

**Metrics:**
- Throughput (tokens/s) across varying (B, L, α)
- Time-to-first-token (TTFT) and inter-token latency (ITL)
- Adaptation speed: tokens to converge after workload shift
- Controller overhead: per-iteration decision time

**Workloads:**
- ShareGPT traces (natural B/L distribution)
- LMSYS-Chat-1M traces (diverse prompt/completion lengths)
- Synthetic sweeps: fixed-L varying B, fixed-B varying L
- Burst workloads: sudden batch size changes to test adaptation

### Multi-GPU Extension

With tensor parallelism (TP), the profiling approach extends naturally — just
profile with the actual multi-GPU setup. The search space stays the same;
only the measured timings change.

```
Multi-GPU (TP=N):
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │
│ D │ T   │ │ D │ T   │ │ D │ T   │ │ D │ T   │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     └───────────┴─ AllReduce ┴───────────┘
```

**Draft model placement strategies:**

| Strategy | When to use |
|----------|-------------|
| Sharded (like target) | Large draft model |
| Replicated per GPU | Small draft model |
| Subset of GPUs | Heterogeneous setups |

### Why Coarse-Grained (Not Kernel-Level)

We adjust per-iteration, not per-kernel:

1. **Temporal misalignment**: Draft and target kernels don't execute in lockstep
2. **Overhead**: Per-kernel SM mask changes (~2μs × thousands) exceeds gains
3. **Macro-balance suffices**: T_draft ≈ T_target over iteration is sufficient
4. **Hardware scheduling**: GPU optimizes within each partition

### References

- [BulletServe](https://arxiv.org/abs/2504.19516): Dynamic SM partitioning for prefill/decode
- [libsmctrl](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/): SM masking library
- [SpecInfer](https://arxiv.org/abs/2305.09781): Tree-based speculative inference
- [EAGLE](https://arxiv.org/abs/2401.15077): Feature-aligned draft models

## Dependencies

- CUDA toolkit (for building libsmctrl)
- PyTorch with CUDA support
- SGLang













This is a solid systems research direction. Let me give you structured feedback on the idea, formulation, and a research plan.

## Feedback on the Core Idea

**Strengths for SC26:**

SC values rigorous performance modeling with real systems implementation — your idea hits both. The key insight that **the optimal speculative decoding strategy is workload-dependent** (batch size, context length) and that a unified adaptive system should switch regimes is compelling. The three-regime design (non-colocated → colocated → AR) captures a real tradeoff space that existing serving systems largely ignore.

**The core tension you're exploiting is real:** speculative decoding trades extra compute for fewer serial decoding steps, but as batch size grows, the system becomes memory-bandwidth-bound and the "free" compute for drafting disappears. Colocation with SM partitioning is an intermediate regime that lets you pipeline draft/verify across batch halves.

**Potential concerns to address early:**

1. **Novelty differentiation** — There's substantial recent work on speculative decoding in serving (SpecInfer, Sequoia, Eagle-2, etc.) and on disaggregated/colocated serving (Splitwise, DistServe, Sarathi). You need to clearly articulate what's new. I think the **analytical model for regime boundaries + adaptive switching** is your differentiator, not just the mechanism itself.

2. **SM partitioning mechanism** — How exactly do you partition SMs? NVIDIA MPS? MIG? CUDA streams with resource hints? Each has different overhead and granularity. MPS is most flexible but has interference issues; MIG is cleaner but coarser.

3. **Dual-batch pipeline correctness** — There's a subtle dependency: when draft generates tokens for batch-half A in the next round, it needs to know the accepted tokens from target's verification of A in the previous round. This means you can't fully decouple the pipeline. The actual pipeline is:

```
Stage 1: Draft(A, γ tokens) || Target(B_verify)
Stage 2: Draft(B, γ tokens) || Target(A_verify)
Stage 3: Draft(A', γ tokens) || Target(B'_verify)  ← A' starts from accepted prefix of A
```

This works because A's verification in Stage 2 finishes before A' drafting is needed in Stage 3. But you need to formalize this carefully.

Let me search for the most recent related work to ensure we're not missing anything critical.Now let me put together a comprehensive analysis. Based on my research and understanding of the landscape, here's my detailed feedback:

---

## 1. Assessment: Is This SC-Worthy?

**Short answer: Conditionally yes, but you need to sharpen the novelty angle carefully.**

The landscape has gotten extremely crowded in the last ~6 months. Here's the critical positioning challenge:

**What already exists:**
- **MagicDec** (Chen et al., 2024, ICLR?) analyzes exactly the bottleneck shift you're describing (compute-bound → memory-bound as B×L grows) and proposes selecting drafting strategy based on batch size and sequence length. They even have an analytical cost model with `T_target(B, S)` and `T_draft(B, S)`.
- **Su et al., "The Synergy of Speculative Decoding and Batching"** (2023) directly shows that optimal γ depends on batch size and builds a quantitative model for it.
- **Nexus, Bullet, DuetServe, Semi-PD, PD-Multiplexing** (all 2025) — These are all doing intra-GPU SM partitioning via MPS or CUDA GreenContext, but for **prefill vs. decode disaggregation**, not draft vs. target.

**What does NOT exist (your novelty):**
Nobody has applied SM-level spatial partitioning to **colocate draft and target models** with pipelined dual-batch execution, nor jointly optimized (γ, SM_ratio) as a 2D optimization problem with regime switching. This is genuinely new and combines two threads of work that haven't been connected.

**My honest assessment:** This is more naturally an **MLSys, ASPLOS, or ISCA** paper than SC. SC's strength is large-scale HPC, and your contribution is primarily about single-node GPU resource optimization for inference. That said, SC has accepted LLM serving papers, and if you frame it as a performance modeling and systems optimization paper with rigorous analytical modeling validated at scale, it can work. Consider whether MLSys 2026 (deadline likely September 2025 — too soon?) or ASPLOS 2027 could be alternatives.

---

## 2. Problem Formulation

Let me help you formalize this properly. Here's the framework:

### Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| L | Sequence (context) length |
| γ | Speculation length (draft tokens per round) |
| r | SM ratio allocated to target (0 < r < 1) |
| α | Per-token acceptance rate |
| T_T(B, L, r) | Target model single-step decode latency |
| T_D(B, L, r) | Draft model single-token generation latency |

### Non-Colocated Speculative Decoding

Draft and target each get 100% of their GPUs (separate GPUs or sequential on same GPU).

**Time per speculative round (sequential):**
```
T_round = γ · T_D(B, L, 1.0) + T_T(B, L+γ, 1.0)
```

**Expected accepted tokens per round** (geometric acceptance model):
```
E[accepted] = (1 - α^(γ+1)) / (1 - α)
```

**Per-token latency:**
```
t_non_coloc(B, L, γ) = T_round / E[accepted]
```

**Optimization:**
```
γ* = argmin_γ  t_non_coloc(B, L, γ)
```

### Colocated Speculative Decoding (Dual-Batch Pipeline)

Split batch into two halves. Pipeline:
- **Stage k**: Target verifies half A (from round k-1) using r·SMs ‖ Draft generates γ tokens for half B using (1-r)·SMs
- **Stage k+1**: Target verifies half B ‖ Draft generates for half A

**Stage latency:**
```
T_stage = max(T_T(B/2, L+γ, r), γ · T_D(B/2, L, 1-r))
```

But critically, you **must model memory bandwidth contention**. Both target verification and draft generation are memory-bandwidth-bound during decode. Sharing the same HBM means:
```
T_stage ≠ max(T_T, T_D)  in general
T_stage ≈ max(T_T, T_D) + Δ_contention(r, B, L)
```

This contention term is what Nexus, DuetServe, and others spend significant effort modeling. You need this too.

**Per-token latency (full batch, 2 pipeline stages):**
```
t_coloc(B, L, γ, r) = (2 · T_stage) / E[accepted]    [for full batch B]
```

Equivalently, throughput = B · E[accepted] / (2 · T_stage)

**Optimization:**
```
(γ*, r*) = argmin_{γ,r}  t_coloc(B, L, γ, r)
```

### Autoregressive Baseline
```
t_AR(B, L) = T_T(B, L, 1.0)
```

### Regime Selection

For given (B, L), the serving system selects:
```
mode* = argmin { t_non_coloc(B, L, γ*_nc),  t_coloc(B, L, γ*_c, r*_c),  t_AR(B, L) }
```

Your hypothesis is that the regime boundaries form regions in (B, L) space:
- Small B, short L → Non-colocated SD wins
- Medium B or long L → Colocated SD wins
- Very large B, short L → AR wins

### Critical Missing Piece: The Latency Model

You need `T_T(B, L, r)` and `T_D(B, L, r)` as analytical or empirical functions. The standard roofline-based decomposition is:

```
T(B, L, r) = max(T_compute(B, r), T_memory(B, L))
```

Where:
- `T_compute(B, r) ∝ FLOPs / (r · peak_FLOPS)` — scales with 1/r
- `T_memory(B, L) ∝ (W_params + B · KV_cache(L)) / BW_mem` — **does NOT scale with r** since all SM partitions share the same HBM bandwidth

This is the fundamental reason your idea works: in memory-bound regimes, giving fewer SMs to a model barely slows it down, because the bottleneck is HBM bandwidth, not compute. You're exploiting the "wasted" compute capability to run the draft model.

But it also means memory bandwidth contention is your biggest enemy in colocated mode.

---

## 3. Critical Related Work You Must Differentiate From

**Tier 1 — Must cite and differentiate deeply:**
1. **MagicDec** (Chen et al., 2024) — Closest to your analytical framing. They model cost ratios and optimal γ. You must show your formulation generalizes theirs by adding the SM ratio dimension and colocation.
2. **Su et al., "Synergy of Speculative Decoding and Batching"** (2023) — Your work extends this by adding the SM ratio and colocation dimensions.
3. **Nexus** (Shi et al., 2025) — SM partitioning for prefill/decode. You apply the same mechanism but for draft/target.
4. **DuetServe** (2025) and **Bullet / Semi-PD** — Same SM partitioning mechanism, different application.

**Tier 2 — Must cite:**
5. **EXSpec / "Batch Speculative Decoding Done Right"** (Zhang et al., 2025) — Shows batch SD is broken in production systems. Motivates your work.
6. **SPIRe** (MatX) — Performance model for throughput of speculative decoding with different draft models.
7. **Speculative Speculative Decoding (SSD)** — Runs drafter and verifier in parallel on separate hardware, pre-computing multiple verification outcomes.
8. **NanoFlow** (Zhu et al., 2024), **Liger** — Kernel-level parallelism overlapping compute/memory/network.

---

## 4. Key Technical Challenges

1. **Memory bandwidth contention model** — This is the hardest part. You need to model how shared HBM bandwidth degrades both target and draft performance when they run concurrently. Nexus uses a contention-aware analytical model; you should study their approach.

2. **SM partitioning mechanism** — CUDA GreenContext (available on Hopper+) is the cleanest option. MPS is more widely available (Ampere+) but has overhead. MIG is too coarse. I'd recommend targeting GreenContext on H100 as primary and MPS as fallback.

3. **Pipeline correctness** — The dual-batch pipeline has a subtle data dependency: draft for batch-half A in round k+1 needs the verified tokens from round k. You need to show that the pipeline staging is correct and doesn't add an extra round of latency.

4. **KV cache management** — Both target and draft need KV caches. In colocated mode, this is extra memory pressure. You need to account for this in your model.

5. **Continuous batching integration** — Real serving systems use continuous batching where batch composition changes every iteration. Your dual-batch split needs to handle this gracefully.

---

## 5. Concrete Research Plan (Priority Order)

### Phase 1: Foundations (Weeks 1-3)
**Do these first:**

1. **Deep-read the core papers:** MagicDec, Su et al. (Synergy), Nexus, DuetServe. Understand their latency models in detail. Map exactly where your contribution differs.

2. **Profile baseline latency models.** On a single H100 (or A100):
   - Measure `T_target(B, L)` and `T_draft(B, L)` for representative model pairs (e.g., Llama-3.1-70B / Llama-3.1-8B, or Llama-3.1-8B / Llama-3.2-1B)
   - Vary B ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}, L ∈ {512, 1K, 2K, 4K, 8K, 16K, 32K}
   - Identify the compute-bound → memory-bound crossover points

3. **Prototype SM partitioning.** Set up MPS or GreenContext and measure:
   - `T_target(B, L, r)` for r ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
   - `T_draft(B, L, 1-r)` concurrently
   - Quantify the memory bandwidth contention overhead Δ

### Phase 2: Analytical Model (Weeks 3-5)

4. **Build the latency model** with contention. Fit an analytical model:
   ```
   T(B, L, r) = max(c₁·FLOPs/(r·F_peak), c₂·(Params + B·KV(L))/BW_eff(r))
   ```
   where `BW_eff(r)` captures the reduced effective bandwidth under contention.

5. **Derive optimal (γ*, r*)** analytically or via fast grid search. Show the regime boundaries in (B, L) space.

6. **Validate the model** against real measurements. This is what makes the paper strong.

### Phase 3: System Implementation (Weeks 5-9)

7. **Implement the adaptive serving system.** Build on SGLang or vLLM:
   - Mode selector: given (B, L), choose AR / non-colocated SD / colocated SD
   - Dual-batch pipeline engine for colocated mode
   - Dynamic γ and r adjustment

8. **Implement the fallback mechanism.** When speculative decoding fails (low acceptance rate), seamlessly switch to AR. This needs to be low-overhead.

### Phase 4: Evaluation (Weeks 9-12)

9. **End-to-end evaluation:**
   - Baselines: vLLM (AR), SGLang (chunked prefill), vLLM with spec decoding, non-colocated SD
   - Workloads: Mix of short/long context, varying arrival rates
   - Metrics: Throughput (tokens/s), per-token latency, TTFT, TBT, SLO attainment
   - Hardware: H100 80GB (GreenContext), A100 80GB (MPS)
   - Models: Llama-3.1-70B/8B, Llama-3.1-8B/1B, possibly a 405B with TP

10. **Ablation studies:**
    - Benefit of adaptive (γ, r) vs. fixed
    - Benefit of regime switching vs. single mode
    - Sensitivity to acceptance rate
    - Overhead of mode switching

---

## 6. What to Do Literally Right Now

In priority order:

1. **Read MagicDec and Su et al. cover-to-cover.** Understand their cost models. Your analytical contribution needs to clearly subsume and extend theirs.

2. **Set up profiling infrastructure.** Get a single-GPU profiling setup where you can measure decode latency at various (B, L) points. This is your empirical foundation.

3. **Prototype SM partitioning.** Run a toy experiment: two processes on one GPU via MPS, one running target decode, one running draft decode. Measure whether you get meaningful overlap and what the contention overhead is. **This is your feasibility gate** — if contention overhead is too high, the whole idea needs rethinking.

4. **Write a 2-page mini-proposal** articulating exactly what your contribution is vs. MagicDec + Nexus. If you can't write a crisp 2-paragraph differentiation, the reviewers will have the same problem.

The feasibility experiment (point 3) is the most important early milestone. If the SM-partitioned colocation shows meaningful speedup for the target (B, L) regime you're hypothesizing, you have a paper. If contention kills the gains, you need to pivot (perhaps to temporal interleaving rather than spatial partitioning).

Would you like me to help you formalize any specific part of this — the analytical model, the experiment design, or the writing structure?