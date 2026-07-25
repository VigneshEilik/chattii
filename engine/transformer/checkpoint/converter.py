from typing import Dict

import torch


class CheckpointConverter:
    """
    Converts model state dictionaries to and from external weight schemas.
    """

    @staticmethod
    def to_flat(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transforms nested parameter labels into flat double-underscore representations.
        """
        return {key.replace(".", "__"): val for key, val in state_dict.items()}

    @staticmethod
    def from_flat(flat_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transforms flat double-underscore labels back into standard nested dot format.
        """
        return {key.replace("__", "."): val for key, val in flat_state_dict.items()}
