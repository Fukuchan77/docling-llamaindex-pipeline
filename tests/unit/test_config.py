"""Settings defaults and fail-fast validation (spec §6)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from docling_llamaindex_pipeline.config import EmbeddingProvider, LLMProvider, get_settings

if TYPE_CHECKING:
    from tests.conftest import SettingsFactory


def test_defaults_are_local_ollama(settings_factory: SettingsFactory) -> None:
    settings = settings_factory()

    assert settings.embedding_provider is EmbeddingProvider.OLLAMA
    assert settings.llm_provider is LLMProvider.OLLAMA
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.api_host == "127.0.0.1"
    assert settings.cors_allow_origins == []
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_reads_environment(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_factory()  # clears the managed env first
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://app.example.com"]')
    monkeypatch.setenv("TEMP_UPLOAD_DIR", "/srv/uploads")
    get_settings.cache_clear()
    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.embedding_provider is EmbeddingProvider.OPENAI
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-env"
    assert settings.cors_allow_origins == ["https://app.example.com"]
    assert settings.temp_upload_dir == Path("/srv/uploads")


def test_secrets_are_masked_in_repr(settings_factory: SettingsFactory) -> None:
    settings = settings_factory(openai_api_key="sk-very-secret", llm_provider="openai")

    assert "sk-very-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding_provider": "openai"}, "OPENAI_API_KEY required"),
        ({"llm_provider": "openai"}, "OPENAI_API_KEY required"),
        ({"embedding_provider": "ibm"}, "WATSONX_APIKEY, WATSONX_PROJECT_ID required"),
        ({"embedding_provider": "ibm", "watsonx_apikey": "k"}, "WATSONX_PROJECT_ID required"),
        (
            {"app_env": "production", "cors_allow_origins": ["*"], "admin_api_key": "k"},
            "wildcard",
        ),
        ({"app_env": "production"}, "ADMIN_API_KEY is required in production"),
    ],
)
def test_misconfiguration_fails_fast(
    settings_factory: SettingsFactory, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        settings_factory(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_provider": "cohere"},
        {"max_upload_size_mb": 0},
        {"query_timeout_seconds": 0},
        {"max_concurrent_ingestions": 0},
        {"docling_device": "tpu"},
    ],
)
def test_out_of_range_values_are_rejected(
    settings_factory: SettingsFactory, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        settings_factory(**overrides)


def test_production_with_explicit_origins_and_admin_key_is_valid(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        app_env="production",
        cors_allow_origins=["https://app.example.com"],
        admin_api_key="k",
    )

    assert settings.app_env == "production"


def test_empty_env_values_mean_unset(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_factory()
    for name in ("ADMIN_API_KEY", "OPENAI_API_KEY", "WATSONX_PROJECT_ID", "DOCLING_ARTIFACTS_PATH"):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.admin_api_key is None
    assert settings.openai_api_key is None
    assert settings.watsonx_project_id is None
    assert settings.docling_artifacts_path is None
