from typing import Any, Iterator, Optional

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class HuggingFaceLoader(BaseLoader):
    """Streams records from a HuggingFace Hub dataset in streaming (iterable) mode."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.name = name
        self.split = split
        self.text_column = text_column
        self.kwargs = kwargs

    def stream(self) -> Iterator[TextSample]:
        try:
            import datasets
        except ImportError:
            raise ImportError(
                "HuggingFaceLoader requires the 'datasets' package. "
                "Please install it using: pip install datasets"
            )

        # Load in streaming mode to avoid RAM saturation
        ds = datasets.load_dataset(
            self.path,
            name=self.name,
            split=self.split,
            streaming=True,
            **self.kwargs,
        )

        for idx, item in enumerate(ds):
            if not isinstance(item, dict):
                text = str(item)
                meta = {"index": idx, "source": self.path}
            else:
                text = item.get(self.text_column, "")
                meta = {k: v for k, v in item.items() if k != self.text_column}
                meta.update({"index": idx, "source": self.path})

            if text:
                yield TextSample(
                    text=str(text),
                    sample_id=f"hf_{idx}",
                    metadata=meta,
                )
