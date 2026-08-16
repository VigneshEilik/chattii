from fastapi import APIRouter

from engine.platform.auth.user import get_user_repository
from engine.platform.usage.tracker import get_usage_tracker

router = APIRouter(prefix="/v1/admin", tags=["Admin Dashboard"])


@router.get("/dashboard")
def get_admin_dashboard():
    users_count = len(get_user_repository().list_users())
    usage_summary = get_usage_tracker().get_summary()

    return {
        "system_health": {
            "status": "healthy",
            "cpu_utilization_pct": 42.0,
            "ram_utilization_pct": 61.0,
            "gpu_utilization_pct": 73.0,
            "vram_utilization_pct": 68.0,
        },
        "performance_metrics": {
            "active_requests": 3,
            "requests_per_min": usage_summary["total_requests"],
            "avg_ttft_ms": usage_summary["avg_ttft_ms"],
            "p95_latency_sec": 1.45,
            "cache_hit_rate_pct": 82.5,
        },
        "users_count": users_count,
        "usage": usage_summary,
    }


@router.get("/models")
def get_admin_models():
    return [
        {
            "model_id": "chatii-v1",
            "status": "Loaded",
            "vram_gb": 3.8,
            "requests": 12832,
            "tps": 42.7,
            "context_length": 4096,
        }
    ]
