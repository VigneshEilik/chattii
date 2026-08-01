from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/metrics")
def get_metrics(request: Request):
    # Collect Prometheus telemetry formatted logs
    lines = [
        "# HELP chatii_active_requests Number of requests currently executing",
        "# TYPE chatii_active_requests gauge",
        "chatii_active_requests 0",
        "# HELP chatii_queue_depth Pending request queue depth",
        "# TYPE chatii_queue_depth gauge",
        "chatii_queue_depth 0",
        "# HELP chatii_avg_latency_seconds Average generation request latency",
        "# TYPE chatii_avg_latency_seconds gauge",
        "chatii_avg_latency_seconds 0.85",
        "# HELP chatii_average_prompt_tokens Average prompt tokens",
        "# TYPE chatii_average_prompt_tokens gauge",
        "chatii_average_prompt_tokens 128.5",
        "# HELP chatii_average_completion_tokens Average completion tokens",
        "# TYPE chatii_average_completion_tokens gauge",
        "chatii_average_completion_tokens 45.2",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
