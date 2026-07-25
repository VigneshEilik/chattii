import torch

from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel


def main() -> None:
    """
    Runs an end-to-end forward pass demonstration on synthetic sequences.
    """
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)

    batch_size = 2
    seq_len = 128

    print(f"Input Shape:\n[{batch_size},{seq_len}]")
    print("\nv\n")

    print(f"Embedding:\n[{batch_size},{seq_len},{config.d_model}]")
    print("\nv\n")

    print(f"Decoder Block x {config.num_layers}")
    print("\nv\n")

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    _ = model(x)

    print(f"Output:\n[{batch_size},{seq_len},{config.vocab_size}]")
    print("without errors.")


if __name__ == "__main__":
    main()
