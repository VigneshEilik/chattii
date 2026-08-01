from engine.inference.batching.dynamic import DynamicBatcher
from engine.inference.batching.request_queue import RequestQueue
from engine.inference.request import GenerationRequest


def test_dynamic_batcher():
    batcher = DynamicBatcher(max_batch_size=2)
    reqs = [GenerationRequest("p1"), GenerationRequest("p2"), GenerationRequest("p3")]
    batches = batcher.merge_requests(reqs)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_request_queue_and_priority_scheduler():
    queue = RequestQueue()
    r1 = GenerationRequest("p1", priority=1)
    r2 = GenerationRequest("p2", priority=10)
    r3 = GenerationRequest("p3", priority=5)

    queue.enqueue(r1)
    queue.enqueue(r2)
    queue.enqueue(r3)

    batch = queue.get_next_batch(2)
    assert len(batch) == 2
    # Highest priority first (r2 has 10, r3 has 5)
    assert batch[0] == r2
    assert batch[1] == r3
