class ModelMetadata:
    """Dataclass holding static properties and constraints for registered models."""

    def __init__(
        self,
        model_id: str,
        version: str,
        description: str,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
    ) -> None:
        self.model_id = model_id
        self.version = version
        self.description = description
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers

    def to_dict(self) -> dict:
        return {
            "id": self.model_id,
            "object": "model",
            "created": 1720000000,
            "owned_by": "chatii",
            "version": self.version,
            "description": self.description,
            "max_seq_len": self.max_seq_len,
        }
