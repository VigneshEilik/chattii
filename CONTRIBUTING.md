# Contributing to Chatii

Thank you for your interest in contributing to Chatii! To maintain high code quality and architectural integrity as Chatii grows, we enforce several engineering standards.

---

## 🏛️ Architectural Standards (Operating System Model)

Chatii follows a strict **Operating System design philosophy**:
1. **Decoupled Interfaces**: Stable production interfaces reside in `core/interfaces/`. These classes should change very rarely.
2. **Pluggable Engines**: Specific implementations (e.g., FlashAttention, custom BPE Tokenizer) reside in `engine/` or other dedicated directories and must subclass their respective base interface.
3. **No Hidden Dependencies**: Modules must not import concrete engine implementations directly. Always depend on abstractions defined in `core/interfaces/` and inject dependencies at runtime.

---

## 📝 Coding Standards

1. **Python Version**: All code must target **Python 3.11**.
2. **Linting and Formatting**: We use `ruff` to manage code style. Before submitting a PR, run:
   ```bash
   ruff check .
   ruff format .
   ```
3. **Type Hints**: Type hinting is required for all public functions, classes, and fields.

---

## 🧪 Testing Policy

**Never merge code without tests.**
- Unit tests go in `tests/unit/`.
- Integration tests go in `tests/integration/`.
- Performance/benchmarking tests go in `tests/performance/`.

To run the test suite locally, use `pytest`:
```bash
pytest
```

---

## 📄 Architectural Decision Records (ADRs)

For any major architectural change or design decision (e.g., choice of tokenization strategy, model topology, database schema changes), you must document it in `docs/adr/`.
- Create a file named `ADR-XXX.md` (e.g., `ADR-001.md`).
- Outline the **Context**, **Decision**, and **Consequences** of the change.

---

## 🛠️ Development Workflow

1. Fork the repository and create your branch from `main`.
2. Ensure your local environment is configured using `.env.development`.
3. Verify your changes pass all unit tests and `ruff` checks.
4. Open a Pull Request detailing the changes, and link any related issues.
