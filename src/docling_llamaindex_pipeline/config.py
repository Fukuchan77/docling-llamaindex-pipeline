"""Runtime configuration loaded from environment variables (spec §6.1).

A frozen :class:`pydantic_settings.BaseSettings` view of the deployment
environment. Misconfiguration fails fast at construction time, i.e. at app
startup, rather than on the first request:

* missing credentials for the *default* embedding / LLM provider;
* a wildcard CORS origin when ``APP_ENV=production`` (spec §6.3.3);
* a missing ``ADMIN_API_KEY`` when ``APP_ENV=production`` (the embedding
  hot-swap endpoint wipes the index, so it must not be left open).

This module is the only place in ``src/`` allowed to carry default model IDs
(``tests/unit/test_app.py::test_default_model_ids_live_only_in_config``
enforces that); every other module reads them from :class:`Settings`.

Model policy: every default is an IBM Granite model (generation, embeddings,
the chunking tokenizer, Docling's layout / TableFormer / Granite-Docling
models), and no default selects a model developed in China
(``tests/unit/test_model_policy.py``). That is also why OCR is pinned to an
explicit engine instead of Docling's auto-selection, which picks RapidOCR
(PaddleOCR-derived models) on Linux when it is installed.

``get_settings()`` returns a process-wide singleton. Tests build
:class:`Settings` directly (with ``_env_file=None``) and inject it into
:func:`docling_llamaindex_pipeline.app.create_app` instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["EmbeddingProvider", "LLMProvider", "Settings", "get_settings"]


class EmbeddingProvider(StrEnum):
    """Embedding backends that can be selected at boot or hot-swapped at runtime."""

    OLLAMA = "ollama"
    IBM = "ibm"
    OPENAI = "openai"


class LLMProvider(StrEnum):
    """LLM backends used for response synthesis (spec §2.2 "LLM Engine")."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Frozen typed view of the runtime environment.

    Attributes:
        app_env: ``development`` / ``staging`` / ``production``. ``production``
            enables the stricter CORS and admin-key checks.
        log_level: Standard ``logging`` level name for the package logger.
        api_host: Bind address used by the ``docling-llamaindex-pipeline``
            entry point. Loopback by default; set ``0.0.0.0`` in containers.
        api_port: Bind port used by the entry point.
        embedding_provider: Embedding backend active at boot.
        llm_provider: LLM backend used for answer synthesis.
        ollama_base_url: Ollama daemon base URL (no ``/v1`` suffix).
        ollama_embed_model: Ollama embedding model tag. ``granite-embedding:278m``
            is multilingual (incl. Japanese), 768-dimensional, 512-token context.
        ollama_llm_model: Ollama chat model tag.
        ollama_request_timeout: Seconds the Ollama LLM client waits per call.
        ollama_context_window: ``num_ctx`` sent to Ollama. Left unset, LlamaIndex
            uses the model maximum (131072 for granite4.2), which spills the KV
            cache onto the CPU and makes calls time out.
        watsonx_apikey: IBM Cloud API key (required for ``ibm`` embeddings).
        watsonx_url: watsonx.ai regional endpoint.
        watsonx_project_id: watsonx.ai project ID (required for ``ibm``).
        watsonx_embed_model: watsonx.ai embedding model ID.
        openai_api_key: OpenAI API key (required for any ``openai`` provider).
        openai_embed_model: OpenAI embedding model ID.
        openai_llm_model: OpenAI chat model ID.
        anthropic_api_key: Anthropic API key (required for the ``anthropic`` LLM).
        anthropic_llm_model: Anthropic chat model ID.
        temp_upload_dir: Directory uploads are staged in before parsing. Each
            upload gets its own private sub-directory, removed after parsing.
        max_upload_size_mb: Upper bound on an uploaded file (HTTP 413 above it).
        max_concurrent_ingestions: Docling conversions allowed to run at once.
            Layout / TableFormer models are memory-hungry, so the default is 1.
        query_timeout_seconds: Seconds ``POST /api/v1/query`` may take before
            it is aborted with HTTP 504.
        docling_do_ocr: Run OCR on bitmap content of PDFs / images.
        docling_ocr_engine: OCR engine. ``tesseract`` needs the ``tesseract``
            binary (and its language packs); ``easyocr`` needs the ``easyocr``
            package.
        docling_ocr_lang: OCR languages as ``iso:`` tags, which Docling maps
            to each engine's own codes.
        docling_do_table_structure: Run TableFormer table-structure recovery.
        docling_device: Accelerator for Docling's models.
        docling_artifacts_path: Pre-downloaded Docling model directory. Set it
            for air-gapped deployments so no model is fetched at runtime.
        chunker: ``hybrid`` splits Docling's structural chunks further so
            none exceeds ``chunk_max_tokens`` of ``chunk_tokenizer``;
            ``hierarchical`` keeps one chunk per document element with no
            token limit (no tokenizer download).
        chunk_tokenizer: Hugging Face tokenizer matching the embedding model.
        chunk_max_tokens: Token budget per chunk; keep it at or below the
            embedding model's context, or the server truncates the input.
        cors_allow_origins: Allowed CORS origins. Empty disables CORS.
        admin_api_key: When set, ``POST /api/v1/config/embedding`` requires a
            matching ``X-Admin-Token`` header.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `KEY=` (as in .env.example) means "unset", not "empty string":
        # otherwise an empty ADMIN_API_KEY would lock the admin route.
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    embedding_provider: EmbeddingProvider = EmbeddingProvider.OLLAMA
    llm_provider: LLMProvider = LLMProvider.OLLAMA

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "granite-embedding:278m"
    ollama_llm_model: str = "granite4.2:latest"
    ollama_request_timeout: float = Field(default=120.0, gt=0)
    ollama_context_window: int = Field(default=8192, gt=0)

    watsonx_apikey: SecretStr | None = None
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_project_id: str | None = None
    watsonx_embed_model: str = "ibm/slate-125m-english-rtrvr-v2"

    openai_api_key: SecretStr | None = None
    openai_embed_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    anthropic_api_key: SecretStr | None = None
    anthropic_llm_model: str = "claude-opus-5-5"

    temp_upload_dir: Path = Path("./data_uploads")
    max_upload_size_mb: int = Field(default=50, ge=1)
    max_concurrent_ingestions: int = Field(default=1, ge=1)
    query_timeout_seconds: float = Field(default=120.0, gt=0)

    docling_do_ocr: bool = True
    docling_ocr_engine: Literal["tesseract", "easyocr"] = "tesseract"
    docling_ocr_lang: list[str] = Field(default_factory=lambda: ["iso:jpn", "iso:eng"])
    docling_do_table_structure: bool = True
    docling_device: Literal["auto", "cpu", "cuda", "mps", "xpu"] = "auto"
    docling_artifacts_path: Path | None = None

    chunker: Literal["hybrid", "hierarchical"] = "hybrid"
    chunk_tokenizer: str = "ibm-granite/granite-embedding-278m-multilingual"
    chunk_max_tokens: int = Field(default=512, ge=16)

    cors_allow_origins: list[str] = Field(default_factory=list[str])
    admin_api_key: SecretStr | None = None

    @property
    def max_upload_bytes(self) -> int:
        """``max_upload_size_mb`` expressed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def missing_credentials(self, provider: EmbeddingProvider | LLMProvider) -> list[str]:
        """Return the env var names a provider needs but that are unset.

        Shared by boot-time validation and the runtime hot-swap endpoint so
        both report the same variable names.

        Args:
            provider: An embedding or LLM provider.

        Returns:
            Names of the missing environment variables; empty when complete.
        """
        required: dict[str, object] = {}
        if provider == EmbeddingProvider.IBM:
            required = {
                "WATSONX_APIKEY": self.watsonx_apikey,
                "WATSONX_PROJECT_ID": self.watsonx_project_id,
            }
        elif provider in (EmbeddingProvider.OPENAI, LLMProvider.OPENAI):
            required = {"OPENAI_API_KEY": self.openai_api_key}
        elif provider == LLMProvider.ANTHROPIC:
            required = {"ANTHROPIC_API_KEY": self.anthropic_api_key}
        return [name for name, value in required.items() if not value]

    @model_validator(mode="after")
    def _check_constraints(self) -> Settings:
        """Enforce cross-field invariants.

        Raises:
            ValueError: On missing provider credentials or an unsafe
                production configuration. Pydantic wraps it into a
                ``ValidationError``.
        """
        for provider in (self.embedding_provider, self.llm_provider):
            missing = self.missing_credentials(provider)
            if missing:
                msg = f"{', '.join(missing)} required when provider '{provider}' is selected."
                raise ValueError(msg)

        if self.app_env == "production":
            if "*" in self.cors_allow_origins:
                msg = (
                    "CORS_ALLOW_ORIGINS must list explicit origins in production; "
                    "the '*' wildcard is not allowed."
                )
                raise ValueError(msg)
            if self.admin_api_key is None:
                msg = "ADMIN_API_KEY is required in production (protects /api/v1/config/embedding)."
                raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()
