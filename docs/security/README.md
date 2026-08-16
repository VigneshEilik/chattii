# Chatii AI Safety & Guardrails

Chatii enforces multi-tier safety checks before and after inference.

## Guardrail Pipeline
1. **Prompt Injection Inspection**: Scans for system bypass attempts.
2. **PII Masking**: Redacts emails, SSNs, and credit cards.
3. **Execution Sandboxing**: Validates tool invocation boundaries.
