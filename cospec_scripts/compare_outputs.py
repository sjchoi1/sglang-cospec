"""Compare outputs between standalone and colocated speculative decoding."""
import requests
import json
import sys

URL = "http://localhost:30000"

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of exercise?",
    "Describe the process of photosynthesis.",
    "What is machine learning?",
    "Tell me about the history of the internet.",
    "How does a rocket engine work?",
    "What are prime numbers?",
    "Explain the theory of relativity.",
]

def run(output_file):
    import concurrent.futures

    def send_prompt(prompt):
        resp = requests.post(
            f"{URL}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
            },
        )
        return prompt, resp.json()

    # Send all prompts concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as executor:
        futures = {executor.submit(send_prompt, p): p for p in PROMPTS}
        for future in concurrent.futures.as_completed(futures):
            prompt, data = future.result()
            results.append({"prompt": prompt, "text": data["text"]})
            print(f"Prompt: {prompt[:50]}")
            print(f"Output: {data['text'][:100]}...")
            print()

    # Sort by prompt for stable comparison
    results.sort(key=lambda x: x["prompt"])

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "outputs.json"
    run(output_file)
