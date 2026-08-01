import os
import subprocess
import sys
from typing import List, Optional

from engine.distributed.launcher.rendezvous import DistributedRendezvous


class DistributedLauncher:
    """Spawns process group workers on single or multi-node systems (simulating torchrun)."""

    @staticmethod
    def launch(
        script_path: str,
        script_args: Optional[List[str]] = None,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: Optional[int] = None,
    ) -> List[subprocess.Popen]:
        script_args = script_args or []
        master_port = master_port or DistributedRendezvous.find_free_port()

        processes = []
        for rank in range(world_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            env["WORLD_SIZE"] = str(world_size)
            env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = str(master_port)

            cmd = [sys.executable, script_path] + script_args
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)

        return processes
