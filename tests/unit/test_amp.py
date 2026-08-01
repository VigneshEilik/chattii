import torch

from engine.distributed.optimization.amp import AMPAutocastContext
from engine.distributed.optimization.scaler import DistributedGradScaler


def test_amp_autocast_and_scaler() -> None:
    scaler = DistributedGradScaler(enabled=True)
    loss = torch.tensor([4.0], requires_grad=True)
    scaled = scaler.scale(loss)

    if torch.cuda.is_available():
        # Scaled loss should be larger than original loss if scaled
        assert scaled.item() > loss.item()

    # Verify context
    ctx = AMPAutocastContext.get_autocast("cpu", enabled=True, dtype=torch.bfloat16)
    assert ctx is not None
