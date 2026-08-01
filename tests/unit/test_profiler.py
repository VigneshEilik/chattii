from engine.distributed.profiler.communication import CommunicationProfiler
from engine.distributed.profiler.gpu import GPUProfiler
from engine.distributed.profiler.memory import MemoryProfiler
from engine.distributed.profiler.throughput import ThroughputProfiler
from engine.distributed.profiler.timeline import DistributedTimelineProfiler


def test_profilers() -> None:
    # Throughput
    tp = ThroughputProfiler()
    stats = tp.get_throughput(1024)
    assert stats["tokens_per_second"] >= 0

    # GPU
    gpu_stats = GPUProfiler.profile_gpus()
    assert "gpu_available" in gpu_stats

    # Memory
    mem_stats = MemoryProfiler.get_summary()
    assert mem_stats["current_ram_mb"] >= 0

    # Communication
    comm = CommunicationProfiler()
    comm.record_operation(0.123)
    assert comm.get_summary()["total_communication_seconds"] == 0.123

    # Timeline
    timeline = DistributedTimelineProfiler()
    timeline.record_event("start")
    assert len(timeline.get_timeline()) == 1
