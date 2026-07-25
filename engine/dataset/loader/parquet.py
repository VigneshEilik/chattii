import os
from typing import Iterator

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class ParquetLoader(BaseLoader):
    """Streams rows from a Parquet file by reading row groups sequentially."""

    def __init__(self, path: str) -> None:
        self.path = path

    def stream(self) -> Iterator[TextSample]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Parquet source file not found: {self.path}")

        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "ParquetLoader requires the 'pyarrow' package. "
                "Please install it using: pip install pyarrow"
            )

        parquet_file = pq.ParquetFile(self.path)
        col_names = parquet_file.schema.names

        # Find best column for text content
        text_col = "text"
        if "text" not in col_names:
            if "content" in col_names:
                text_col = "content"
            else:
                text_col = col_names[0]

        for rg_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg_idx)
            df = table.to_pydict()
            texts = df[text_col]

            for row_idx, text in enumerate(texts):
                if text:
                    meta = {
                        "row_group": rg_idx,
                        "row_index": row_idx,
                        "source": self.path,
                    }
                    for col in col_names:
                        if col != text_col:
                            meta[col] = df[col][row_idx]

                    yield TextSample(
                        text=str(text),
                        sample_id=f"parquet_{rg_idx}_{row_idx}",
                        metadata=meta
                    )
