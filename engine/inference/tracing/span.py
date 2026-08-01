import time


class Span:
    """Represents a single profiled execution span with start and end times."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time = 0.0

    def finish(self) -> None:
        self.end_time = time.perf_counter()

    @property
    def duration(self) -> float:
        if self.end_time == 0.0:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
