import pytest

from engine.distributed.backend.registry import BackendRegistry


# Register dummy backend for testing
@BackendRegistry.register("dummy_backend")
def init_dummy() -> None:
    pass


def test_backend_registry() -> None:
    # Get registered backend
    fn = BackendRegistry.get("dummy_backend")
    assert fn is not None

    # Missing backend check
    with pytest.raises(KeyError):
        BackendRegistry.get("non_existent_backend")
