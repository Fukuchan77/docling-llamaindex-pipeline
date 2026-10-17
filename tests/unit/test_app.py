"""App composition: default wiring, CORS (spec §6.3.3), entry point, model-ID hygiene."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from fastapi.testclient import TestClient

import docling_llamaindex_pipeline
from docling_llamaindex_pipeline.app import create_app
from docling_llamaindex_pipeline.config import Settings
from docling_llamaindex_pipeline.parsing import DoclingDocumentParser

if TYPE_CHECKING:
    from tests.conftest import HarnessFactory, SettingsFactory

SRC = Path(docling_llamaindex_pipeline.__file__).parent


def test_default_wiring_boots_without_any_model_server(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(chunker="hierarchical"))

    with TestClient(app) as client:
        health = client.get("/health")
        container = app.state.container

    assert health.status_code == 200
    assert health.json()["embedding_model"] == "granite-embedding:278m"
    assert isinstance(container.parser, DoclingDocumentParser)


def test_unavailable_chunk_tokenizer_fails_startup(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default hybrid chunker needs its tokenizer; when it cannot load, startup fails loudly."""

    def _offline(*_: object, **__: object) -> None:
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(HuggingFaceTokenizer, "from_pretrained", _offline)
    app = create_app(settings_factory())

    with (
        pytest.raises(
            RuntimeError, match=r"(?s)Could not load chunk tokenizer .*CHUNKER=hierarchical"
        ),
        TestClient(app),
    ):
        pass  # pragma: no cover — startup raises first.


def test_cors_disabled_by_default(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = harness.client.get("/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_listed_origins(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(cors_allow_origins=["https://app.example.com"])

    allowed = harness.client.options(
        "/api/v1/query",
        headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST"},
    )
    denied = harness.client.get("/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "access-control-allow-origin" not in denied.headers


def test_cors_wildcard_never_sends_credentials(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(cors_allow_origins=["*"])

    response = harness.client.get("/health", headers={"Origin": "https://any.example"})

    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_non_numeric_content_length_is_rejected(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = harness.client.post(
        "/api/v1/query", content=b"{}", headers={"Content-Length": "abc"}
    )

    assert response.status_code == 413


def test_main_runs_uvicorn_factory(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    from docling_llamaindex_pipeline import config

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _run(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", _run)
    settings = settings_factory(api_host="0.0.0.0", api_port=9000)  # noqa: S104
    monkeypatch.setattr(config, "get_settings", lambda: settings)

    docling_llamaindex_pipeline.main()

    assert calls == [
        (
            ("docling_llamaindex_pipeline.app:create_app",),
            {"factory": True, "host": "0.0.0.0", "port": 9000},  # noqa: S104
        )
    ]


def test_default_model_ids_live_only_in_config() -> None:
    """Model IDs are configuration: only ``config.py`` may name one (via a default)."""
    defaults = {
        str(field.default)
        for name, field in Settings.model_fields.items()
        if name.endswith(("_model", "_embed_model", "_llm_model", "_tokenizer"))
    }
    assert {"granite4.2:latest", "granite-embedding:278m"} <= defaults

    offenders = [
        f"{path.relative_to(SRC)}: {model_id}"
        for path in SRC.rglob("*.py")
        if path.name != "config.py"
        for model_id in defaults
        if re.search(rf"[\"']{re.escape(model_id)}[\"']", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
