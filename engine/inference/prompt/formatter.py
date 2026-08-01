from typing import Any, Dict, List

from engine.inference.prompt.template import PromptTemplate


class ChatFormatter:
    """Converts multi-turn dialogs to token sequences and applies token budget constraints."""

    def __init__(self, tokenizer: Any, template_type: str = "chatml") -> None:
        self.tokenizer = tokenizer
        self.template = PromptTemplate(template_type)

    def format_and_tokenize(
        self, messages: List[Dict[str, str]], max_length: int = 2048
    ) -> List[int]:
        formatted_text = self.template.format(messages)
        # Append target assistant indicator to prompt completion
        if messages and messages[-1].get("role") != "assistant":
            formatted_text += "<|im_start|>assistant\n"

        # Tokenize using tokenizer encode method
        token_ids = self.tokenizer.encode(formatted_text)
        if len(token_ids) > max_length:
            token_ids = token_ids[-max_length:]
        return token_ids
