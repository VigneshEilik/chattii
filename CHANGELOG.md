# Changelog

All notable changes to the Chatii project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0-dev] - 2026-07-18

### Added
- **Directory Freeze**: Created the complete directory skeleton including production interfaces (`core/interfaces/`), engines (`engine/`), model registries, dataset registries, docs, testing wrappers, and workflow setups.
- **Licensing**: Added Apache License 2.0.
- **Environment Management**: Added environment configuration files (`.env`, `.env.example`, `.env.development`, `.env.production`).
- **Configuration Engine**: Implemented Pydantic-backed configuration loader supporting modular YAML files,dotenv priority loading, env overrides (`CHATII_MODEL__HIDDEN_DIM`), and type-safe runtime overrides.
- **Centralized Logger**: Implemented a standard library structured JSON logging formatting utility configured for stdout/file logging.
- **Dockerization Stack**: Created `Dockerfile.dev` and `Dockerfile.prod`, and docker-compose files (`docker-compose.yml`, `docker-compose.dev.yml`) for PostgreSQL and Redis services.
- **Package Specifications**: Configured `pyproject.toml` targeting Python 3.11 with `ruff` and `pytest` parameters, and generated `requirements.txt`.
