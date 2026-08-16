from engine.platform.usage.tracker import UsageTracker


def test_record_usage():
    tracker = UsageTracker()
    rec = tracker.record_usage("usr-1", "key-1", "chatii-v1", 10, 20, 150.0, 500.0)
    assert rec.user_id == "usr-1"
    assert rec.total_tokens == 30
    assert rec.prompt_tokens == 10
    assert rec.completion_tokens == 20


def test_usage_summary_empty():
    tracker = UsageTracker()
    summary = tracker.get_summary()
    assert summary["total_requests"] == 0
    assert summary["total_tokens"] == 0


def test_usage_summary_aggregated():
    tracker = UsageTracker()
    tracker.record_usage("usr-2", "key-1", "chatii-v1", 10, 20, 100.0, 400.0)
    tracker.record_usage("usr-2", "key-1", "chatii-v1", 20, 30, 200.0, 600.0)

    summary = tracker.get_summary("usr-2")
    assert summary["total_requests"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 50
    assert summary["total_tokens"] == 80
    assert summary["avg_ttft_ms"] == 150.0
    assert summary["avg_latency_ms"] == 500.0


def test_usage_summary_user_filtered():
    tracker = UsageTracker()
    tracker.record_usage("usr-A", None, "chatii-v1", 10, 10, 100.0, 200.0)
    tracker.record_usage("usr-B", None, "chatii-v1", 50, 50, 200.0, 400.0)

    summary_a = tracker.get_summary("usr-A")
    assert summary_a["total_requests"] == 1
    assert summary_a["total_tokens"] == 20


def test_usage_record_model_tracking():
    tracker = UsageTracker()
    rec = tracker.record_usage("usr-3", None, "chatii-base", 5, 5, 50.0, 100.0)
    assert rec.model == "chatii-base"


def test_multiple_usage_records_averaging():
    tracker = UsageTracker()
    tracker.record_usage("u", None, "m", 1, 1, 100.0, 200.0)
    tracker.record_usage("u", None, "m", 1, 1, 300.0, 400.0)
    summary = tracker.get_summary("u")
    assert summary["avg_ttft_ms"] == 200.0
    assert summary["avg_latency_ms"] == 300.0


def test_usage_record_id_uniqueness():
    tracker = UsageTracker()
    r1 = tracker.record_usage("u", None, "m", 1, 1, 10.0, 10.0)
    r2 = tracker.record_usage("u", None, "m", 1, 1, 10.0, 10.0)
    assert r1.id != r2.id


def test_usage_summary_global():
    tracker = UsageTracker()
    tracker.record_usage("u1", None, "m", 10, 10, 10.0, 10.0)
    tracker.record_usage("u2", None, "m", 20, 20, 20.0, 20.0)
    global_summary = tracker.get_summary()
    assert global_summary["total_requests"] == 2
    assert global_summary["total_tokens"] == 60


def test_usage_zero_latency():
    tracker = UsageTracker()
    rec = tracker.record_usage("u", None, "m", 0, 0, 0.0, 0.0)
    assert rec.total_latency_ms == 0.0


def test_usage_key_id_logging():
    tracker = UsageTracker()
    rec = tracker.record_usage("u", "key-123", "m", 5, 5, 10.0, 10.0)
    assert rec.key_id == "key-123"
