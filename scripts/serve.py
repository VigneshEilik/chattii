import argparse

import uvicorn

from core.utils.config import ChatiiConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Chatii HTTP API")
    parser.add_argument(
        "--config-dir", default="configs", help="Path to modular configs folder"
    )
    args = parser.parse_args()

    # Load configuration
    config = ChatiiConfig.load_from_dir(args.config_dir)

    print(f"Starting Chatii API gateway at {config.api.host}:{config.api.port}...")

    # Run uvicorn server
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers,
    )


if __name__ == "__main__":
    main()
