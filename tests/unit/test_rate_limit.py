from engine.api.rate_limit.limiter import RateLimiter


def test_rate_limiter_token_bucket():
    cfg = {"limits": {"rate_limit": {"requests_per_minute": 60, "burst": 2}}}
    limiter = RateLimiter(cfg)

    # First two requests allowed
    assert limiter.is_allowed("client1", use_sliding_window=False) is True
    assert limiter.is_allowed("client1", use_sliding_window=False) is True

    # Third request blocked due to burst limit
    assert limiter.is_allowed("client1", use_sliding_window=False) is False


def test_rate_limiter_sliding_window():
    cfg = {"limits": {"rate_limit": {"requests_per_minute": 2, "burst": 2}}}
    limiter = RateLimiter(cfg)

    assert limiter.is_allowed("client2", use_sliding_window=True) is True
    assert limiter.is_allowed("client2", use_sliding_window=True) is True

    # Third blocked
    assert limiter.is_allowed("client2", use_sliding_window=True) is False
