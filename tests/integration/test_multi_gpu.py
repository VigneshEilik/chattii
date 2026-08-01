import os

from engine.distributed.launcher.launcher import DistributedLauncher


def test_multi_gpu_spawning_gloo() -> None:
    script_path = "tests/integration/dummy_gloo_init.py"
    script_content = """import os
import torch.distributed as dist
try:
    dist.init_process_group(backend="gloo")
    print(f"Rank {dist.get_rank()} initialized successfully.")
    dist.destroy_process_group()
except Exception as e:
    print(f"Init failed: {e}")
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        processes = DistributedLauncher.launch(script_path, world_size=2)
        assert len(processes) == 2
        for p in processes:
            p.wait()
            assert p.returncode == 0
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
