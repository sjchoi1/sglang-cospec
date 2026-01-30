"""Plot SM partition sweep results."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("/workspace/sglang/cospec_scripts/sm_partition_results.json") as f:
    data = json.load(f)

results = data["results"]
seq_ms = data["sequential_ms"]
no_part_ms = data["no_partition_ms"]
total_tpcs = data["total_tpcs"]

ratios = [r["ratio"] for r in results]
draft_ms = [r["draft_ms"] for r in results]
target_ms = [r["target_ms"] for r in results]
concurrent_ms = [r["concurrent_ms"] for r in results]
sum_ms = [r["sum_ms"] for r in results]
overlap_pct = [r["overlap_pct"] for r in results]
draft_tpcs = [r["draft_tpcs"] for r in results]
target_tpcs = [r["target_tpcs"] for r in results]

labels = [f"{r['draft_tpcs']}/{r['target_tpcs']}" for r in results]

fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

# --- Top: Timing breakdown ---
ax = axes[0]
x = np.arange(len(ratios))
width = 0.25

bars_draft = ax.bar(x - width, draft_ms, width, label="Draft only", color="#4CAF50", alpha=0.85)
bars_target = ax.bar(x, target_ms, width, label="Target only", color="#2196F3", alpha=0.85)
bars_concurrent = ax.bar(x + width, concurrent_ms, width, label="Concurrent", color="#FF5722", alpha=0.85)

# Reference lines
ax.axhline(y=seq_ms, color="black", linestyle="--", linewidth=1, label=f"Sequential (all SMs) = {seq_ms:.2f}ms")
ax.axhline(y=no_part_ms, color="gray", linestyle=":", linewidth=1, label=f"Concurrent (no partition) = {no_part_ms:.2f}ms")

ax.set_ylabel("Time (ms)")
ax.set_title(f"SM Partition Sweep — Draft vs Target vs Concurrent\n"
             f"GPU: RTX A6000 ({total_tpcs} TPCs), Draft: 8×Linear(2048), Target: 32×Linear(4096), bs=256")
ax.set_xticks(x)
ax.set_xticklabels([f"{r:.0%}\n({l})" for r, l in zip(ratios, labels)])
ax.set_xlabel("Draft SM Ratio (draft/target TPCs)")
ax.legend(loc="upper right")
ax.set_ylim(0, max(concurrent_ms) * 1.15)

# Annotate concurrent bars
for i, (c, s) in enumerate(zip(concurrent_ms, sum_ms)):
    savings = s - c
    if savings > 0.01:
        ax.annotate(f"{savings:.2f}ms\nsaved",
                    xy=(x[i] + width, c), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color="#FF5722")

# --- Bottom: Overlap efficiency ---
ax2 = axes[1]
colors = ["#4CAF50" if o > 50 else "#FFC107" if o > 10 else "#F44336" for o in overlap_pct]
ax2.bar(x, overlap_pct, 0.6, color=colors, alpha=0.85)
ax2.axhline(y=100, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_ylabel("Overlap %")
ax2.set_xlabel("Draft SM Ratio (draft/target TPCs)")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{r:.0%}" for r in ratios])
ax2.set_ylim(min(min(overlap_pct) - 10, -15), max(max(overlap_pct) + 10, 110))
ax2.set_title("Overlap Efficiency (100% = perfect, 0% = none)")

for i, o in enumerate(overlap_pct):
    ax2.annotate(f"{o:.0f}%", xy=(x[i], o), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("/workspace/sglang/cospec_scripts/sm_partition_sweep.png", dpi=150, bbox_inches="tight")
print("Saved to cospec_scripts/sm_partition_sweep.png")
