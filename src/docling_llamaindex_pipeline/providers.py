"""Embedding / LLM model factories (spec §2.2, §4.4).

Builders construct LlamaIndex model adapters from :class:`Settings` without
performing network I/O for Ollama and OpenAI, so the app boots (and
``/health`` answers) while a model server is down; reachability surfaces as
HTTP 503 on the first request that needs it. ``WatsonxEmbeddings`` is the
exception: the IBM SDK authenticates while the client is constructed.

The OpenAI and IBM integrations are optional extras (``uv sync --extra openai``
/ ``--extra ibm``) and are imported lazily, so a default Ollama-only install
never pays for them. A missing extra is reported as a
:class:`ProviderConfigurationError` (HTTP 400 on hot-swap).

Models are always passed explicitly to the index and query engine; the
global ``llama_index.core.Settings`` is never mutated, so a hot-swap cannot
leak into a request that is already running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from docling_llamaindex_pipeline.config import EmbeddingProvider, LLMProvider
from docling_llamaindex_pipeline.errors import ProviderConfigurationError

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM

    from docling_llamaindex_pipeline.config import Settings

__all__ = ["EmbeddingSelection", "build_embedding", "build_llm"]


@dataclass(frozen=True, slots=True)
class EmbeddingSelection:
    """An embedding adapter together with the identity it was built from.

    Attributes:
        provider: The provider the adapter talks to.
        model_name: The provider-side model ID.
        model: The LlamaIndex embedding adapter.
    """

    provider: EmbeddingProvider
    model_name: str
    model: BaseEmbedding


def _require_credentials(settings: Settings, provider: EmbeddingProvider | LLMProvider) -> None:
    missing = settings.missing_credentials(provider)
    if missing:
        msg = f"{', '.join(missing)} must be set to use provider '{provider}'."
        raise ProviderConfigurationError(msg)


def _missing_extra(provider: str, extra: str) -> ProviderConfigurationError:
    return ProviderConfigurationError(
        f"Provider '{provider}' requires the '{extra}' extra: uv sync --extra {extra}"
    )


def build_embedding(
    provider: EmbeddingProvider,
    settings: Settings,
    *,
    model_name: str | None = None,
) -> EmbeddingSelection:
    """Construct the embedding adapter for ``provider``.

    Args:
        provider: Embedding backend to use.
        settings: Source of endpoints, credentials and default model IDs.
        model_name: Overrides the provider's default model from ``settings``.

    Returns:
        The adapter and the provider / model it was built for.

    Raises:
        ProviderConfigurationError: A credential is missing or the provider's
            optional extra is not installed.
    """
    _require_credentials(settings, provider)
    match provider:
        case EmbeddingProvider.OLLAMA:
            name = model_name or settings.ollama_embed_model
            model: BaseEmbedding = OllamaEmbedding(
                model_name=name,
                base_url=settings.ollama_base_url,
            )
        case EmbeddingProvider.OPENAI:
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
            except ImportError as exc:
                raise _missing_extra(provider, "openai") from exc

            assert settings.openai_api_key is not None  # noqa: S101 — checked above.
            name = model_name or settings.openai_embed_model
            model = OpenAIEmbedding(
                model=name,
                api_key=settings.openai_api_key.get_secret_value(),
            )
        case EmbeddingProvider.IBM:
            try:
                from llama_index.embeddings.ibm import WatsonxEmbeddings
            except ImportError as exc:
                raise _missing_extra(provider, "ibm") from exc

            assert settings.watsonx_apikey is not None  # noqa: S101 — checked above.
            name = model_name or settings.watsonx_embed_model
            model = WatsonxEmbeddings(
                model_id=name,
                url=settings.watsonx_url,
                apikey=settings.watsonx_apikey.get_secret_value(),
                project_id=settings.watsonx_project_id,
            )
    return EmbeddingSelection(provider=provider, model_name=name, model=model)


def build_llm(settings: Settings) -> LLM:
    """Construct the synthesis LLM selected by ``settings.llm_provider``.

    Args:
        settings: Source of the provider, endpoint, credentials and model ID.

    Returns:
        The LlamaIndex LLM adapter.

    Raises:
        ProviderConfigurationError: A credential is missing or the provider's
            optional extra is not installed.
    """
    provider = settings.llm_provider
    _require_credentials(settings, provider)
    match provider:
        case LLMProvider.OLLAMA:
            return Ollama(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                request_timeout=settings.ollama_request_timeout,
                context_window=settings.ollama_context_window,
            )
        case LLMProvider.OPENAI:
            try:
                from llama_index.llms.openai import OpenAI
            except ImportError as exc:
                raise _missing_extra(provider, "openai") from exc

            assert settings.openai_api_key is not None  # noqa: S101 — checked above.
            return OpenAI(
                model=settings.openai_llm_model,
                api_key=settings.openai_api_key.get_secret_value(),
            )
        case LLMProvider.ANTHROPIC:
            try:
                from llama_index.llms.anthropic import Anthropic
            except ImportError as exc:
                raise _missing_extra(provider, "anthropic") from exc

            assert settings.anthropic_api_key is not None  # noqa: S101 — checked above.
            return Anthropic(
                model=settings.anthropic_llm_model,
                api_key=settings.anthropic_api_key.get_secret_value(),
            )
