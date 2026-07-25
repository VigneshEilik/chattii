from engine.dataset.base.loader import BaseLoader
from engine.dataset.config import DatasetPipelineConfig


class DatasetFactory:
    """Instantiates streaming data loaders based on format extensions and configuration tags."""

    @staticmethod
    def create_loader(config: DatasetPipelineConfig) -> BaseLoader:
        loader_type = config.loader_type.lower()
        path = config.path

        if loader_type == "txt":
            from engine.dataset.loader.text import TextLoader
            return TextLoader(path, encoding=config.encoding)
        elif loader_type == "json":
            from engine.dataset.loader.json import JsonLoader
            return JsonLoader(path, encoding=config.encoding)
        elif loader_type == "jsonl":
            from engine.dataset.loader.jsonl import JsonlLoader
            return JsonlLoader(path, encoding=config.encoding)
        elif loader_type == "csv":
            from engine.dataset.loader.csv import CsvLoader
            return CsvLoader(path, encoding=config.encoding)
        elif loader_type == "parquet":
            from engine.dataset.loader.parquet import ParquetLoader
            return ParquetLoader(path)
        else:
            raise ValueError(f"Unsupported dataset loader type: {config.loader_type}")
