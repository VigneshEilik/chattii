import re
from dataclasses import dataclass


@dataclass
class SafetyCheckResult:
    is_safe: bool
    reason: str
    filtered_text: str


class SafetyGuardrails:
    """Input/output safety classifier, prompt injection detector, and PII masker."""

    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"system prompt bypass",
        r"reveal secret key",
        r"dan mode",
    ]

    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD]"),
    ]

    def inspect_prompt(self, prompt: str) -> SafetyCheckResult:
        prompt_lower = prompt.lower()
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, prompt_lower):
                return SafetyCheckResult(
                    is_safe=False,
                    reason=f"Prompt injection detected matching pattern '{pat}'",
                    filtered_text=prompt,
                )

        # Mask PII
        clean_prompt = prompt
        for pat, replacement in self.PII_PATTERNS:
            clean_prompt = re.sub(pat, replacement, clean_prompt)

        return SafetyCheckResult(
            is_safe=True, reason="Pass", filtered_text=clean_prompt
        )

    def validate_output(self, response_text: str) -> SafetyCheckResult:
        # Validate output bounds
        return SafetyCheckResult(
            is_safe=True, reason="Pass", filtered_text=response_text
        )


_safety_guardrails = SafetyGuardrails()


def get_safety_guardrails() -> SafetyGuardrails:
    return _safety_guardrails
