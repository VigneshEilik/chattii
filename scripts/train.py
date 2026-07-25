import argparse
import logging

from core.utils.config import ChatiiConfig
from core.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chatii Foundation Model")
    parser.add_argument(
        "--config-dir", default="configs", help="Path to modular configs folder"
    )
    args = parser.parse_args()

    # Load configuration
    config = ChatiiConfig.load_from_dir(args.config_dir)

    # Initialize Logger
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        file_path=config.logging.file_path,
    )

    logger = logging.getLogger("training")
    logger.info(
        "Initializing Chatii modular training process...", extra={"epoch": 0, "step": 0}
    )
    logger.info(
        f"Model selected: {config.model.name} "
        f"({config.model.hidden_dim} dim, {config.model.num_layers} layers)"
    )
    logger.warning("Training engine is currently not implemented in Milestone 0.")


if __name__ == "__main__":
    main()
