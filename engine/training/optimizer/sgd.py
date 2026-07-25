import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """
    Custom implementation of Stochastic Gradient Descent (SGD) from scratch.
    """

    def __init__(
        self, params, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Initialize momentum buffer on-demand
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # Update weights
                p.add_(buf, alpha=-lr)

        return loss
