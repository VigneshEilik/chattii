from engine.distributed.benchmark import DistributedBenchmark


def test_scaling_efficiency() -> None:
    results = DistributedBenchmark.run_benchmark()
    assert results["scaling_efficiency_pct"] > 90.0
    assert results["step_latency_seconds"] >= 0
