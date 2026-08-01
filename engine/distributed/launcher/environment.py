import os
from typing import Dict


class DistributedEnvironment:
    """Retrieves and coordinates standard PyTorch distributed environment settings."""

    @staticmethod
    def get_env() -> Dict[str, str]:
        return {
            "RANK": os.environ.get("RANK", "0"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK", "0"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE", "1"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
        }

    @staticmethod
    def setup_env(
        rank: int, local_rank: int, world_size: int, master_addr: str, master_port: int
    ) -> None:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

    @staticmethod
    def is_distributed() -> bool:
        return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
