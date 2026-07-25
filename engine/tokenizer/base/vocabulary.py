from abc import ABC, abstractmethod
from typing import Dict


class BaseVocabulary(ABC):
    """
    Stable abstract base interface for Chatii vocabulary management.
    """

    @abstractmethod
    def add_token(self, token: str) -> int:
        """
        Adds a token to the vocabulary and returns its mapped integer ID.
        """
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """
        Retrieves the integer ID of a given string token.
        If token is not found, should return the <unk> ID.
        """
        pass

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """
        Retrieves the string token representation of a given integer ID.
        """
        pass

    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """
        Returns registered special token constants.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total count of registered tokens.
        """
        pass
