import socket


class DistributedRendezvous:
    """Manages port discovery and node handshakes."""

    @staticmethod
    def find_free_port() -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return int(port)

    @staticmethod
    def verify_connection(host: str, port: int, timeout: float = 5.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
