import sys

from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def main() -> None:
    print("Initializing Chatii Inference Service...")
    service = InferenceService()
    service.load_model()

    print("\n--- Demonstration: Autoregressive Text Generation ---")
    prompt = "Explain self-attention simply."
    print(f"Prompt: {prompt}")

    config = GenerationConfig(config_dict={"max_new_tokens": 32, "temperature": 0.7})
    response = service.generate(prompt, config=config)
    print(f"Response: {response.text}")
    print(
        f"Generated {response.generated_tokens} tokens in {response.latency_seconds:.3f}s ({response.tokens_per_second:.2f} tokens/sec)"
    )

    print("\n--- Demonstration: Token-by-Token Streaming ---")
    prompt_stream = "What is KV caching?"
    print(f"Prompt: {prompt_stream}")
    print("Stream: ", end="", flush=True)

    for token in service.stream(prompt_stream, config=config):
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n\nDemonstration complete.")


if __name__ == "__main__":
    main()
