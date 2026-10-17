"""Embedding / LLM factories (spec §2.2, §4.4). None of these perform network I/O."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

import pytest
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from docling_llamaindex_pipeline.config import EmbeddingProvider
from docling_llamaindex_pipeline.errors import ProviderConfigurationError
from docling_llamaindex_pipeline.providers import build_embedding, build_llm

if TYPE_CHECKING:
    from tests.conftest import SettingsFactory


def test_ollama_embedding_uses_settings(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(ollama_base_url="http://ollama:11434", ollama_embed_model="m-embed")

    selection = build_embedding(EmbeddingProvider.OLLAMA, settings)

    assert selection.provider is EmbeddingProvider.OLLAMA
    assert selection.model_name == "m-embed"
    assert isinstance(selection.model, OllamaEmbedding)
    assert selection.model.base_url == "http://ollama:11434"


def test_model_name_override_wins(settings_factory: SettingsFactory) -> None:
    selection = build_embedding(EmbeddingProvider.OLLAMA, settings_factory(), model_name="other")

    assert selection.model_name == "other"
    assert selection.model.model_name == "other"


def test_openai_embedding(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(
        openai_api_key="sk-test", openai_embed_model="text-embedding-3-large"
    )

    selection = build_embedding(EmbeddingProvider.OPENAI, settings)

    assert isinstance(selection.model, OpenAIEmbedding)
    assert selection.model_name == "text-embedding-3-large"
    assert selection.model.api_key == "sk-test"


def test_ibm_embedding_passes_credentials(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llama_index.embeddings.ibm as ibm_module

    captured: dict[str, Any] = {}

    class _FakeWatsonx(OllamaEmbedding):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(model_name=kwargs["model_id"])

    monkeypatch.setattr(ibm_module, "WatsonxEmbeddings", _FakeWatsonx)
    settings = settings_factory(watsonx_apikey="ibm-key", watsonx_project_id="proj-1")

    selection = build_embedding(EmbeddingProvider.IBM, settings)

    assert selection.provider is EmbeddingProvider.IBM
    assert captured == {
        "model_id": settings.watsonx_embed_model,
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": "ibm-key",
        "project_id": "proj-1",
    }


@pytest.mark.parametrize(
    ("provider", "module", "extra"),
    [
        (EmbeddingProvider.OPENAI, "llama_index.embeddings.openai", "openai"),
        (EmbeddingProvider.IBM, "llama_index.embeddings.ibm", "ibm"),
    ],
)
def test_missing_extra_is_reported(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    provider: EmbeddingProvider,
    module: str,
    extra: str,
) -> None:
    _block_import(monkeypatch, module)
    settings = settings_factory(openai_api_key="k", watsonx_apikey="k", watsonx_project_id="p")

    with pytest.raises(ProviderConfigurationError, match=f"uv sync --extra {extra}"):
        build_embedding(provider, settings)


def test_missing_credentials_are_reported(settings_factory: SettingsFactory) -> None:
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY must be set"):
        build_embedding(EmbeddingProvider.OPENAI, settings_factory())


def test_ollama_llm(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(
        ollama_llm_model="m-chat", ollama_request_timeout=30, ollama_context_window=4096
    )

    llm = build_llm(settings)

    assert isinstance(llm, Ollama)
    assert llm.model == "m-chat"
    assert llm.request_timeout == 30
    assert llm.context_window == 4096


def test_openai_llm(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(llm_provider="openai", openai_api_key="sk", openai_llm_model="c")

    llm = build_llm(settings)

    assert isinstance(llm, OpenAI)
    assert llm.model == "c"


def test_openai_llm_missing_extra(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_factory(llm_provider="openai", openai_api_key="sk")
    _block_import(monkeypatch, "llama_index.llms.openai")

    with pytest.raises(ProviderConfigurationError, match="--extra openai"):
        build_llm(settings)


def _block_import(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_anthropic_llm(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(
        llm_provider="anthropic", anthropic_api_key="sk-ant", anthropic_llm_model="c"
    )

    llm = build_llm(settings)

    assert isinstance(llm, Anthropic)
    assert llm.model == "c"


def test_anthropic_llm_missing_extra(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_factory(llm_provider="anthropic", anthropic_api_key="sk-ant")
    _block_import(monkeypatch, "llama_index.llms.anthropic")

    with pytest.raises(ProviderConfigurationError, match="--extra anthropic"):
        build_llm(settings)
