from typing import Any, Dict, Optional, Type

from engine.dataset.augmentation.hooks import AugmentationHook, ChatPromptFormatter


class AugmentationRegistry:
    """Registry mapping identifiers to specific data augmentation/formatting hooks."""

    _hooks: Dict[str, Type[AugmentationHook]] = {
        "chat_format": ChatPromptFormatter,
    }

    @classmethod
    def register(cls, name: str, hook_cls: Type[AugmentationHook]) -> None:
        cls._hooks[name] = hook_cls

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Optional[AugmentationHook]:
        hook_cls = cls._hooks.get(name)
        if hook_cls:
            return hook_cls(**kwargs)
        return None
