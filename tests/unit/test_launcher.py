import os

from engine.distributed.launcher.environment import DistributedEnvironment
from engine.distributed.launcher.launcher import DistributedLauncher


def test_distributed_environment() -> None:
    DistributedEnvironment.setup_env(1, 1, 2, "127.0.0.1", 29500)
    env = DistributedEnvironment.get_env()
    assert env["RANK"] == "1"
    assert env["LOCAL_RANK"] == "1"
    assert env["WORLD_SIZE"] == "2"
    assert DistributedEnvironment.is_distributed()


def test_distributed_launcher() -> None:
    # Use a simple python snippet that exits successfully
    script_content = "import sys; sys.exit(0)"
    script_path = "tests/unit/dummy_script.py"

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
