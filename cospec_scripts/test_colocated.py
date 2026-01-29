"""Quick test for colocated speculative decoding."""
import os
import time

os.environ["COLOCATED_MIN_BATCH_SIZE"] = "1"


def main():
    import sglang as sgl

    llm = sgl.Engine(
        model_path="Qwen/Qwen3-8B",
        speculative_algorithm="COLOCATED",
        speculative_draft_model_path="Qwen/Qwen3-0.6B",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=3,
        mem_fraction_static=0.50,
        cuda_graph_max_bs=4,
        log_level="error",
    )

    print("Engine ready", flush=True)

    # Test: Multiple requests simultaneously
    prompts = ["What is 2+2?", "Hello world", "Explain gravity briefly."]
    outputs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 32})
    for i, o in enumerate(outputs):
        print(f"Output {i}: {repr(o['text'][:100])}", flush=True)

    llm.shutdown()
    print("All tests passed!", flush=True)


if __name__ == "__main__":
    main()
