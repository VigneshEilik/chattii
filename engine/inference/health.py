from typing import Any, Dict

from engine.inference.service import InferenceService


class InferenceHealthCheck:
    """Diagnoses engine model states and triggers execution warmups."""

    def __init__(self, service: InferenceService) -> None:
        self.service = service

    def check_health(self) -> Dict[str, Any]:
        report = {
            "model_loaded": self.service.model is not None,
            "tokenizer_loaded": self.service.tokenizer is not None,
            "engine_initialized": self.service.engine is not None,
            "status": "healthy",
        }

        if not report["model_loaded"] or not report["tokenizer_loaded"]:
            report["status"] = "unhealthy"

        return report
