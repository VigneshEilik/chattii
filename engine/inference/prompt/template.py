from typing import Dict, List


class PromptTemplate:
    """Applies role markers and system prompts onto user inputs."""

    def __init__(self, template_type: str = "chatml") -> None:
        self.template_type = template_type

    def format(self, messages: List[Dict[str, str]]) -> str:
        # Formats conversation based on ChatML format
        # <|im_start|>system\n...<|im_end|>\n
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(result)
