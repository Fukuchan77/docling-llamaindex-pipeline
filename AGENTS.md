# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Status
- Early scaffold: [`src/docling_llamaindex_pipeline/__init__.py`](src/docling_llamaindex_pipeline/__init__.py) only contains a placeholder `main()`. No FastAPI app, ingestion pipeline, tests, or lint config exist yet. README describes the intent: multi-format document ingestion + RAG API (Docling + LlamaIndex + Ollama embeddings).

## Commands
- Toolchain is pinned in [`mise.toml`](mise.toml) (Python 3.14, uv 0.13); `requires-python >=3.14`. Use `uv`, not pip. There are NO `mise` tasks defined (`mise run gate` does not exist).
- Setup: `uv sync` (dev tools are in the `dev` dependency group: ruff, pyright, ty, pytest, pytest-asyncio, pytest-cov, pip-audit, pre-commit).
- Lint/format: `uv run ruff check .` / `uv run ruff format .` (no `[tool.ruff]` config yet, so defaults apply).
- Type-check: `uv run pyright` and/or `uv run ty check` (both installed, neither configured).
- Tests: `uv run pytest` ; single test: `uv run pytest path/to/test_file.py::test_name -x`. No `[tool.pytest.ini_options]` exists, so set `asyncio_mode` there (or mark tests with `@pytest.mark.asyncio`) when adding async tests.
- Entry point: `uv run docling-llamaindex-pipeline` (script maps to `docling_llamaindex_pipeline:main`).

## Layout / gotchas
- `src/` layout with `uv_build` backend: the package directory name must match the project name with underscores (`docling_llamaindex_pipeline`).
- Settings are expected via `pydantic-settings` (dependency present); `.env` is gitignored.
- `.claude/` and `.serena/` are gitignored local AI-tool folders; do not commit them.
