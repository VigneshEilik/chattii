from engine.dataset.base.dataset import TextSample


class AugmentationHook:
    """Base class for text data formatting, conversation structuring, or RLHF annotations."""

    def augment(self, sample: TextSample) -> TextSample:
        """Applies transformation hook to a sample."""
        return sample


class ChatPromptFormatter(AugmentationHook):
    """Wraps text in standard conversation instructions (e.g. system/user/assistant context)."""

    def __init__(self, template: str = "User: {text}\nAssistant:") -> None:
        self.template = template

    def augment(self, sample: TextSample) -> TextSample:
        formatted = self.template.format(text=sample.text)
        return TextSample(
            text=formatted,
            sample_id=sample.sample_id,
            metadata=sample.metadata.copy(),
        )
