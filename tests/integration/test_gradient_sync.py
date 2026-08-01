import os

from engine.distributed.launcher.launcher import DistributedLauncher


def test_gradient_synchronization_flow() -> None:
    script_path = "tests/integration/dummy_grad_sync.py"
    script_content = """import torch
import torch.nn as nn
import torch.distributed as dist
from engine.distributed.synchronization.gradients import GradientSynchronizer

try:
    dist.init_process_group(backend="gloo")
    model = nn.Linear(2, 2)
    # Set different gradients on each rank
    rank = dist.get_rank()
    for param in model.parameters():
        param.grad = torch.ones_like(param) * (rank + 1)

    # Sync
    GradientSynchronizer.synchronize(list(model.parameters()))

    # Verify - averaged gradients should be 1.5 for ranks [1, 2]
    for param in model.parameters():
        assert torch.allclose(param.grad, torch.ones_like(param) * 1.5)

    print("Grad sync check passed.")
    dist.destroy_process_group()
except Exception as e:
    print(f"Failed: {e}")
    import sys
    sys.exit(1)
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
