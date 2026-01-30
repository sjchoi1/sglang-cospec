"""Test SM partitioning: measure overlap between draft and target streams.

For each draft SM ratio, measures:
  - Draft-only time (restricted SMs)
  - Target-only time (restricted SMs)
  - Concurrent time (both streams)
  - Overlap efficiency = (draft_only + target_only - concurrent) / min(draft_only, target_only)

If overlap is working, concurrent < draft_only + target_only.
"""

import sys
import json

import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/sglang")
from python.sglang.srt.utils.sm_controller import SMController

WARMUP = 20
ITERS = 200


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    device = "cuda"
    dtype = torch.float16

    n_draft_layers = 8
    n_target_layers = 32
    draft = nn.Sequential(
        *[nn.Linear(2048, 2048, bias=False) for _ in range(n_draft_layers)]
    ).to(device, dtype)
    target = nn.Sequential(
        *[nn.Linear(4096, 4096, bias=False) for _ in range(n_target_layers)]
    ).to(device, dtype)

    x_draft = torch.randn(256, 2048, device=device, dtype=dtype)
    x_target = torch.randn(256, 4096, device=device, dtype=dtype)

    sm_ctrl = SMController()
    if not sm_ctrl.enabled:
        print("SM partitioning not available.")
        return

    total_tpcs = sm_ctrl.total_tpcs
    print(f"GPU: {torch.cuda.get_device_name()}, TPCs: {total_tpcs}")
    print(f"Draft: {n_draft_layers}x Linear(2048), Target: {n_target_layers}x Linear(4096), bs=256\n")

    # Baseline: sequential on default stream (all SMs)
    seq_ms = bench(lambda: (draft(x_draft), target(x_target)))

    # No partition concurrent
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    def run_no_part():
        with torch.cuda.stream(s_a):
            draft(x_draft)
        with torch.cuda.stream(s_b):
            target(x_target)
    no_part_ms = bench(run_no_part)

    print(f"{'Sequential (all SMs)':<40} {seq_ms:>8.3f} ms")
    print(f"{'Concurrent (no partition)':<40} {no_part_ms:>8.3f} ms")
    print()

    header = f"{'Ratio':>6} {'Draft TPCs':>10} {'Draft(ms)':>10} {'Target(ms)':>11} {'Sum(ms)':>9} {'Concurrent':>11} {'Overlap%':>9}"
    print(header)
    print("-" * len(header))

    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for ratio in ratios:
        draft_tpcs = max(1, int(total_tpcs * ratio))
        target_tpcs = total_tpcs - draft_tpcs

        s_draft = torch.cuda.Stream()
        s_target = torch.cuda.Stream()
        sm_ctrl.set_stream_mask(s_draft, 0, draft_tpcs)
        sm_ctrl.set_stream_mask(s_target, draft_tpcs, total_tpcs)

        def run_draft_only():
            with torch.cuda.stream(s_draft):
                draft(x_draft)
        def run_target_only():
            with torch.cuda.stream(s_target):
                target(x_target)
        def run_concurrent():
            with torch.cuda.stream(s_draft):
                draft(x_draft)
            with torch.cuda.stream(s_target):
                target(x_target)

        draft_ms = bench(run_draft_only)
        target_ms = bench(run_target_only)
        concurrent_ms = bench(run_concurrent)

        sum_ms = draft_ms + target_ms
        overlap_pct = (sum_ms - concurrent_ms) / min(draft_ms, target_ms) * 100

        results.append({
            "ratio": ratio,
            "draft_tpcs": draft_tpcs,
            "target_tpcs": target_tpcs,
            "draft_ms": draft_ms,
            "target_ms": target_ms,
            "sum_ms": sum_ms,
            "concurrent_ms": concurrent_ms,
            "overlap_pct": overlap_pct,
        })

        print(f"{ratio:>6.0%} {draft_tpcs:>5}/{target_tpcs:<4} {draft_ms:>10.3f} {target_ms:>11.3f} {sum_ms:>9.3f} {concurrent_ms:>11.3f} {overlap_pct:>8.1f}%")

    print()
    print(f"Overlap% = (draft_only + target_only - concurrent) / min(draft, target) * 100")
    print(f"100% = perfect overlap (concurrent = max(draft, target))")
    print(f"  0% = no overlap (concurrent = draft + target)")

    # Save for plotting
    out = {
        "sequential_ms": seq_ms,
        "no_partition_ms": no_part_ms,
        "total_tpcs": total_tpcs,
        "results": results,
    }
    with open("/workspace/sglang/cospec_scripts/sm_partition_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to cospec_scripts/sm_partition_results.json")


if __name__ == "__main__":
    main()
