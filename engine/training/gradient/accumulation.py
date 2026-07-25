class GradientAccumulator:
    """Tracks step counters to trigger optimizer updates after N batches."""

    def __init__(self, accumulation_steps: int) -> None:
        self.accumulation_steps = max(1, accumulation_steps)
        self.current_step = 0

    def should_step(self) -> bool:
        """Returns True if accumulated gradients should trigger an optimizer update."""
        return (self.current_step + 1) % self.accumulation_steps == 0

    def increment(self) -> None:
        """Increments internal accumulation batch step count."""
        self.current_step += 1

    def reset(self) -> None:
        """Resets step count back to 0."""
        self.current_step = 0
