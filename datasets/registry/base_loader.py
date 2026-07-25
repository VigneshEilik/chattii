from abc import ABC, abstractmethod
from typing import Any, Dict, Generator


class BaseTextLoader(ABC):
    """
    Abstract Base Class defining text ingestion interfaces for vocabulary builders
    and tokenization datasets.
    """

    @abstractmethod
    def load_lines(self, file_path: str) -> Generator[str, None, None]:
        """
        Yields normalized string lines from a target file.
        """
        pass

    @abstractmethod
    def get_provenance_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Gathers dataset properties (hash, size, line count, language)
        to populate the vocabulary metadata repository.
        """
        pass
