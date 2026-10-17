# AGENTS.md

Guidance for agents working in this repository. The README covers usage; this
file covers commands, invariants, and the "why" behind non-obvious choices.

## Commands
- Toolchain is pinned in `mise.toml` (Python 3.14, uv). Use `uv`, never pip.
- Setup: `mise run setup` (= `uv sync --all-extras --all-groups && uv run pre-commit install`).
  The `openai` / `ibm` extras are needed for the full test suite.
- Single gate: `mise run check` (lint, format, typecheck, coverage-gated tests). CI runs the same
  tasks; add new checks to `mise.toml`, not only to CI or a hook.
- Single test: `uv run pytest tests/unit/test_api_query.py::test_query_times_out_with_504 -x`.
- Type checkers: `uv run pyright` (strict) and `uv run ty check`, both over `src` and `tests`.
- Live lane: `mise run test:integration` (needs Ollama with the configured models; the PDF case
  also needs Docling layout models).

## Invariants (enforced by tests)
- Unit tests are hermetic: `HF_HUB_OFFLINE=1` (pyproject `env`), no model server. Use the fakes in
  `tests/support/` (`KeywordEmbedding`, `FixedLLM`, `FakeParser`) and inject them through
  `create_app(container_factory=...)` — see `tests/conftest.py::harness_factory`.
- Model IDs appear only as defaults in `config.py` (`tests/unit/test_app.py::test_default_model_ids_live_only_in_config`).
- Model policy: defaults are IBM Granite (`granite4.2:latest`, `granite-embedding:278m`); no model
  developed in China may appear in defaults, `src/`/`examples/` code strings, `.env.example`,
  `mise.toml` or CI (`tests/unit/test_model_policy.py`). OCR is pinned (`DOCLING_OCR_ENGINE`)
  because Docling's auto mode can pick PaddleOCR-derived RapidOCR; the VLM preset is pinned to
  Granite-Docling for the same reason.
- `DeprecationWarning` is an error under pytest; any ignore must be scoped to its emitting module
  with a comment.
- Coverage ratchet: `fail_under = 98`.

## Design notes
- `app.py`, not `main.py`: the package exposes a `main()` console entry point, and a `main`
  submodule would shadow it as a package attribute.
- Never mutate `llama_index.core.Settings`; pass `embed_model` / `llm` explicitly so a hot-swap
  cannot leak into a running request.
- `RagService` serialises ingestion and hot-swap with one `asyncio.Lock`; queries run lock-free on
  a snapshot of the current index.
- Domain errors subclass `errors.PipelineError` (which carries the HTTP status); routes raise them
  and never build error responses themselves. Network failures are detected via the exception
  cause chain (`is_model_server_unreachable`) and map to 503.
- FastAPI evaluates route annotations at runtime, so `api/**` is exempt from ruff's TC00x rules.
- `starlette.testclient` needs `httpx2` (dev dependency) for typed, warning-free tests.
- `mise run audit` ignores PYSEC-2026-3740 (nltk, unreachable, no fix release) — review by 2026-11-05.
- Chunking defaults to `HybridChunker` with the embedding model's HF tokenizer (512 tokens for
  `granite-embedding:278m`), loaded at startup; unit tests use `CHUNKER=hierarchical` or a fake
  `BaseTokenizer`. Startup fails loudly if the tokenizer cannot be loaded.
- `examples/` are runnable docs referenced by `docs/guide/`; ruff + pyright + ty check them.

## Layout / gotchas
- `src/` layout with the `uv_build` backend: package dir must be `docling_llamaindex_pipeline`.
- `.env` is read by `Settings`; empty values (`KEY=`) mean unset (`env_ignore_empty`).
- `.claude/` and `.serena/` are gitignored local AI-tool folders; do not commit them.
