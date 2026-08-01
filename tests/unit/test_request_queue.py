from engine.inference.batching.request_queue import RequestQueue
from engine.inference.request import GenerationRequest


def test_request_queue_lifecycle():
    queue = RequestQueue()
    req = GenerationRequest("hello", session_id="s1")
    queue.enqueue(req)
    assert queue.size() == 1

    queue.cancel("s1")
    batch = queue.get_next_batch(1)
    assert len(batch) == 0
    assert queue.size() == 0
