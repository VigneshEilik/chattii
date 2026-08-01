from engine.api.model_manager.manager import ModelManager
from engine.api.model_registry.registry import ModelRegistry


def test_model_registry_and_manager():
    registry = ModelRegistry()
    assert len(registry.list_models()) > 0
    assert registry.get_metadata("chatii-v1") is not None
    assert registry.get_metadata("invalid-model") is None

    cfg = {
        "models": {
            "default_model": "chatii-v1",
            "model_aliases": {"latest": "chatii-v1"},
        }
    }
    mgr = ModelManager(cfg)
    assert len(mgr.loaded_models) == 1

    service = mgr.get_model("latest")
    assert service is not None
