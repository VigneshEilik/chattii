from typing import Any, Dict, List, Tuple

import torch

from engine.transformer.nn.parameters import Parameter


class Module:
    """
    Base class for all neural network building blocks in Chatii.
    """

    def __init__(self) -> None:
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training: bool = True

    def register_parameter(self, name: str, param: Parameter) -> None:
        """
        Registers a learnable parameter in the module.
        """
        if not isinstance(param, Parameter):
            raise TypeError(
                f"Object registered as '{name}' must be a Parameter instance."
            )
        self._parameters[name] = param
        setattr(self, name, param)

    def register_module(self, name: str, module: "Module") -> None:
        """
        Registers a child submodule.
        """
        if not isinstance(module, Module):
            raise TypeError(f"Object registered as '{name}' must be a Module instance.")
        self._modules[name] = module
        setattr(self, name, module)

    def train(self, mode: bool = True) -> "Module":
        """
        Sets the module and child submodules to training mode.
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """
        Sets the module and child submodules to evaluation mode.
        """
        return self.train(False)

    def parameters(self) -> List[Parameter]:
        """
        Returns an aggregated list of all learnable parameters in this module.
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
        """
        Returns a list of name-tensor tuples for all parameters in this module.
        """
        params = []
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            params.append((key, param.data))
        for name, module in self._modules.items():
            params.extend(module.named_parameters(prefix=f"{prefix}{name}."))
        return params

    def state_dict(self, prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing references to all parameters in this module.
        """
        d = {}
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            d[key] = param.data
        for name, module in self._modules.items():
            d.update(module.state_dict(prefix=f"{prefix}{name}."))
        return d

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], prefix: str = ""
    ) -> None:
        """
        Loads parameters into this module and its submodules from the state dictionary.
        """
        with torch.no_grad():
            for name, param in self._parameters.items():
                key = f"{prefix}{name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])
        for name, module in self._modules.items():
            module.load_state_dict(state_dict, prefix=f"{prefix}{name}.")

    def to(self, device: torch.device) -> "Module":
        """
        Moves all learnable parameters and submodules to the target compute device.
        """
        for param in self._parameters.values():
            param.data = param.data.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the forward computation step. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of Module must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
