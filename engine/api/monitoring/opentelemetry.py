from typing import Any


class OpenTelemetryTracer:
    """Orchestrates OpenTelemetry spans and exports trace details."""

    def __init__(self, service_name: str = "chatii-service") -> None:
        self.service_name = service_name

    def start_span(self, name: str) -> Any:
        # Mock span object for tracing
        return MockSpan(name)


class MockSpan:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "MockSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
