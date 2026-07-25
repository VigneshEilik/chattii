class FlopsCalculator:
    """Calculates model parameter step FLOPs and active TFLOP throughput."""

    @staticmethod
    def calculate_flops_per_step(
        num_parameters: int, batch_size: int, seq_len: int
    ) -> int:
        """
        Calculates total step operations: 6 * P * B * T (for forward + backward passes).
        """
        return 6 * num_parameters * batch_size * seq_len

    @staticmethod
    def calculate_tflops(
        num_parameters: int, batch_size: int, seq_len: int, step_time_sec: float
    ) -> float:
        """
        Returns execution speed of active training loop in TeraFLOPs (TFLOPs).
        """
        if step_time_sec <= 0:
            return 0.0
        flops = FlopsCalculator.calculate_flops_per_step(
            num_parameters, batch_size, seq_len
        )
        return float(flops / (step_time_sec * 1e12))
