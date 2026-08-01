import socket
from typing import Any, Dict


class ClusterNode:
    """Represents a single node machine within the distributed cluster."""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        hostname = socket.gethostname()
        try:
            ip_addr = socket.gethostbyname(hostname)
        except Exception:
            ip_addr = "127.0.0.1"

        return {
            "hostname": hostname,
            "ip_address": ip_addr,
        }
