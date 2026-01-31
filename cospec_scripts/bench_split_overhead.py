"""
Batch Splitting Overhead: FFN vs Attention vs Full Transformer Layer

Plot 1: FFN splitting overhead vs B
Plot 2: Attention splitting overhead vs B (per L)
Plot 3: Full transformer layer splitting overhead vs B (per L) — uses actual model weights

Usage:
  python bench_split_overhead.py --model meta-llama/Meta-Llama-3-8B
  python bench_split_overhead.py --model meta-llama/Meta-Llama-3-8B --save fig_split_overhead.png
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM

# ── Helpers ──────────────────────────────────────────────────

def make_ffn_params(config, device, dtype):
    h = config.hidden_size
    inter = config.intermediate_size
    gate = torch.randn(inter, h, device=device, dtype=dtype)
    up = torch.randn(inter, h, device=device, dtype=dtype)
    down = torch.randn(h, inter, device=device, dtype=dtype)
    return gate, up, down


def ffn_forward(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def make_attn_params(config, device, dtype):
    h = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = h // n_heads
    qkv_dim = h + 2 * n_kv * head_dim
    qkv = torch.randn(qkv_dim, h, device=device, dtype=dtype)
    o = torch.randn(h, h, device=device, dtype=dtype)
    return qkv, o, n_heads, n_kv, head_dim


def attn_forward(x, kv_cache_k, kv_cache_v, qkv, o, n_heads, n_kv, head_dim):
    B = x.shape[0]
    proj = F.linear(x, qkv)
    h = n_heads * head_dim
    q = proj[:, :, :h].view(B, 1, n_heads, head_dim).transpose(1, 2)
    k_new = proj[:, :, h:h + n_kv * head_dim].view(B, 1, n_kv, head_dim).transpose(1, 2)
    v_new = proj[:, :, h + n_kv * head_dim:].view(B, 1, n_kv, head_dim).transpose(1, 2)
    k = torch.cat([kv_cache_k, k_new], dim=2)
    v = torch.cat([kv_cache_v, v_new], dim=2)
    rep = n_heads // n_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn_out = attn_out.transpose(1, 2).reshape(B, 1, h)
    return F.linear(attn_out, o)


# ── Benchmark ────────────────────────────────────────────────

def bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times)


# ── Per-component benchmarks ─────────────────────────────────

def run_ffn_bench(config, B, device, dtype):
    gate, up, down = make_ffn_params(config, device, dtype)
    h = config.hidden_size
    x_full = torch.randn(B, 1, h, device=device, dtype=dtype)
    t_unsplit = bench(lambda: ffn_forward(x_full, gate, up, down))
    x_half = torch.randn(B // 2, 1, h, device=device, dtype=dtype)
    t_split = bench(lambda: (ffn_forward(x_half, gate, up, down),
                              ffn_forward(x_half, gate, up, down)))
    return t_unsplit, t_split


def run_attn_bench(config, B, L, device, dtype):
    qkv, o, n_heads, n_kv, head_dim = make_attn_params(config, device, dtype)
    h = config.hidden_size
    x_full = torch.randn(B, 1, h, device=device, dtype=dtype)
    kv_k = torch.randn(B, n_kv, L, head_dim, device=device, dtype=dtype)
    kv_v = torch.randn(B, n_kv, L, head_dim, device=device, dtype=dtype)
    t_unsplit = bench(lambda: attn_forward(x_full, kv_k, kv_v, qkv, o, n_heads, n_kv, head_dim))
    x_half = torch.randn(B // 2, 1, h, device=device, dtype=dtype)
    kv_k_half = torch.randn(B // 2, n_kv, L, head_dim, device=device, dtype=dtype)
    kv_v_half = torch.randn(B // 2, n_kv, L, head_dim, device=device, dtype=dtype)
    t_split = bench(lambda: (
        attn_forward(x_half, kv_k_half, kv_v_half, qkv, o, n_heads, n_kv, head_dim),
        attn_forward(x_half, kv_k_half, kv_v_half, qkv, o, n_heads, n_kv, head_dim),
    ))
    return t_unsplit, t_split


# ── Full transformer layer benchmark ─────────────────────────

def run_full_layer_bench(model, layer_idx, config, B, L, device, dtype):
    """Run a real transformer layer from the model, unsplit vs split."""
    layer = model.model.layers[layer_idx]
    rotary_emb = model.model.rotary_emb
    h = config.hidden_size
    n_kv = config.num_key_value_heads
    head_dim = h // config.num_attention_heads

    from transformers.cache_utils import DynamicCache

    def make_cache(bs):
        cache = DynamicCache()
        k = torch.randn(bs, n_kv, L, head_dim, device=device, dtype=dtype)
        v = torch.randn(bs, n_kv, L, head_dim, device=device, dtype=dtype)
        cache.update(k, v, layer_idx)
        return cache

    def make_pos_emb(bs):
        position_ids = torch.arange(L, L + 1, device=device).unsqueeze(0).expand(bs, -1)
        # rotary_emb expects hidden_states for dtype/device, and position_ids
        dummy = torch.randn(bs, 1, h, device=device, dtype=dtype)
        cos, sin = rotary_emb(dummy, position_ids)
        return position_ids, (cos, sin)

    # Unsplit: full batch
    x_full = torch.randn(B, 1, h, device=device, dtype=dtype)
    cache_full = make_cache(B)
    pos_ids_full, pos_emb_full = make_pos_emb(B)
    t_unsplit = bench(lambda: layer(
        x_full, position_ids=pos_ids_full, past_key_value=cache_full,
        use_cache=True, position_embeddings=pos_emb_full,
    ))

    # Split: half batch, twice
    x_half = torch.randn(B // 2, 1, h, device=device, dtype=dtype)
    cache_half = make_cache(B // 2)
    pos_ids_half, pos_emb_half = make_pos_emb(B // 2)
    t_split = bench(lambda: (
        layer(x_half, position_ids=pos_ids_half, past_key_value=cache_half,
              use_cache=True, position_embeddings=pos_emb_half),
        layer(x_half, position_ids=pos_ids_half, past_key_value=cache_half,
              use_cache=True, position_embeddings=pos_emb_half),
    ))
    return t_unsplit, t_split


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--save", default="fig_split_overhead.png")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--layer-idx", type=int, default=0, help="Which transformer layer to benchmark")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading config: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    print(f"  hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}")
    print(f"  n_heads={config.num_attention_heads}, n_kv_heads={config.num_key_value_heads}")

    print(f"\nLoading model for full-layer benchmark...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    batch_sizes = list(range(32, 257, 32))  # [32, 64, 96, ..., 256]
    seq_lengths = [256, 512, 1024, 2048]

    # ── 1. FFN splitting overhead ──
    print("\n=== FFN Splitting Overhead ===")
    ffn_overhead = {}
    for B in batch_sizes:
        t_un, t_sp = run_ffn_bench(config, B, device, dtype)
        ratio = t_sp / t_un
        ffn_overhead[B] = ratio
        print(f"  B={B:4d}  unsplit={t_un:.3f}ms  split={t_sp:.3f}ms  overhead={ratio:.2f}x")

    # ── 2. Attention splitting overhead ──
    print("\n=== Attention Splitting Overhead ===")
    attn_overhead = {}
    for L in seq_lengths:
        print(f"\n  L={L}:")
        for B in batch_sizes:
            try:
                t_un, t_sp = run_attn_bench(config, B, L, device, dtype)
                ratio = t_sp / t_un
                attn_overhead[(B, L)] = ratio
                print(f"    B={B:4d}  unsplit={t_un:.3f}ms  split={t_sp:.3f}ms  overhead={ratio:.2f}x")
            except torch.cuda.OutOfMemoryError:
                print(f"    B={B:4d}  OOM, skipping")
                torch.cuda.empty_cache()

    # ── 3. Full transformer layer splitting overhead ──
    print(f"\n=== Full Layer (layer {args.layer_idx}) Splitting Overhead ===")
    full_overhead = {}
    with torch.no_grad():
        for L in seq_lengths:
            print(f"\n  L={L}:")
            for B in batch_sizes:
                try:
                    t_un, t_sp = run_full_layer_bench(
                        model, args.layer_idx, config, B, L, device, dtype)
                    ratio = t_sp / t_un
                    full_overhead[(B, L)] = ratio
                    print(f"    B={B:4d}  unsplit={t_un:.3f}ms  split={t_sp:.3f}ms  overhead={ratio:.2f}x")
                except torch.cuda.OutOfMemoryError:
                    print(f"    B={B:4d}  OOM, skipping")
                    torch.cuda.empty_cache()

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {256: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c", 2048: "#d62728"}

    # Plot 1: FFN overhead vs B
    ax = axes[0]
    bs = sorted(ffn_overhead.keys())
    ax.plot(bs, [ffn_overhead[b] for b in bs], "o-", color="gray", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("(a) FFN Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dual-Batch / Single-Batch Latency", fontsize=12)
    ax.set_ylim(0.8, 2.2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs], fontsize=9)

    # Plot 2: Attention overhead vs B, per L
    ax = axes[1]
    for L in seq_lengths:
        bs_l = sorted([b for (b, l) in attn_overhead if l == L])
        if bs_l:
            vals = [attn_overhead[(b, L)] for b in bs_l]
            ax.plot(bs_l, vals, "o-", label=f"L={L}", color=colors[L],
                    linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("(b) Attention Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dual-Batch / Single-Batch Latency", fontsize=12)
    ax.set_ylim(0.8, 2.2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=9)

    # Plot 3: Full layer overhead vs B, per L
    ax = axes[2]
    for L in seq_lengths:
        bs_l = sorted([b for (b, l) in full_overhead if l == L])
        if bs_l:
            vals = [full_overhead[(b, L)] for b in bs_l]
            ax.plot(bs_l, vals, "o-", label=f"L={L}", color=colors[L],
                    linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("(c) Transformer Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dual-Batch / Single-Batch Latency", fontsize=12)
    ax.set_ylim(0.8, 2.2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=9)

    # Common legend on top
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(seq_lengths),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig(args.save, bbox_inches="tight", dpi=150)
    print(f"\nSaved figure to {args.save}")


if __name__ == "__main__":
    main()
