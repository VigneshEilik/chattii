from typing import Any, Dict

from engine.distributed.launcher.environment import DistributedEnvironment
from engine.distributed.launcher.rendezvous import DistributedRendezvous


class DistributedValidator:
    """Audits cluster layout configurations and connections before starting training."""

    @staticmethod
    def validate_setup() -> Dict[str, Any]:
        env = DistributedEnvironment.get_env()
        # Verify master node accessibility
        port_accessible = DistributedRendezvous.verify_connection(
            env["MASTER_ADDR"], int(env["MASTER_PORT"]), timeout=2.0
        )

        return {
            "master_addr": env["MASTER_ADDR"],
            "master_port": env["MASTER_PORT"],
            "master_accessible": port_accessible
            or env["MASTER_ADDR"] in ["127.0.0.1", "localhost"],
            "is_distributed": DistributedEnvironment.is_distributed(),
        }
