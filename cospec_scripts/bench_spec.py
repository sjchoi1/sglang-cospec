"""Basic benchmark for speculative decoding."""

import argparse
import time
import sglang as sgl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--algorithm", type=str, default="STANDALONE")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is machine learning?",
        "Explain the theory of relativity.",
        "How do computers work?",
        "What causes earthquakes?",
        "Describe how the internet works.",
    ][:args.num_prompts]

    print(f"Target: {args.target_model}")
    print(f"Draft: {args.draft_model}")
    print(f"Algorithm: {args.algorithm}")
    print()

    llm = sgl.Engine(
        model_path=args.target_model,
        speculative_algorithm=args.algorithm,
        speculative_draft_model_path=args.draft_model,
        speculative_num_steps=args.num_steps,
        speculative_eagle_topk=args.topk,
        speculative_num_draft_tokens=args.num_draft_tokens,
    )

    sampling_params = {"temperature": 0, "max_new_tokens": args.max_tokens}

    # Warmup
    print("Warmup...")
    _ = llm.generate(prompts[:1], sampling_params)

    # Benchmark
    print("Running benchmark...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o["meta_info"]["completion_tokens"]) for o in outputs)

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens / elapsed:.2f} tokens/s")

    llm.shutdown()


if __name__ == "__main__":
    main()
