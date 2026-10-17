"""``POST /api/v1/config/embedding`` (spec §4.4) and ``GET /health`` (spec §7.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.support.fakes import FakeParser, text_node

if TYPE_CHECKING:
    import httpx2

    from tests.conftest import Harness, HarnessFactory


def _ingest(harness: Harness) -> None:
    response = harness.client.post("/api/v1/ingest", files={"file": ("a.pdf", b"%PDF")})
    assert response.status_code == 201


def _switch(
    harness: Harness, headers: dict[str, str] | None = None, **body: Any
) -> httpx2.Response:
    return harness.client.post("/api/v1/config/embedding", json=body, headers=headers)


def test_health_reports_counts_and_provider(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([text_node("a"), text_node("b")]))

    before = harness.client.get("/health").json()
    _ingest(harness)
    after = harness.client.get("/health").json()

    assert before == {
        "status": "healthy",
        "indexed_documents": 0,
        "indexed_nodes": 0,
        "active_embedding_provider": "ollama",
        "embedding_model": "keyword-hash",
    }
    assert (after["indexed_documents"], after["indexed_nodes"]) == (1, 2)


def test_switch_to_openai_resets_index(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([text_node("a")]), openai_api_key="sk-test")
    _ingest(harness)

    response = _switch(harness, provider="openai")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UPDATED",
        "active_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "message": "Embedding model changed successfully. Please re-index documents.",
    }
    health = harness.client.get("/health").json()
    assert health["indexed_documents"] == 0
    assert health["active_embedding_provider"] == "openai"
    assert harness.client.post("/api/v1/query", json={"query": "a"}).status_code == 400


def test_switch_honours_model_name_override(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = _switch(harness, provider="ollama", model_name="nomic-embed-text")

    assert response.json()["embedding_model"] == "nomic-embed-text"
    assert harness.client.get("/health").json()["embedding_model"] == "nomic-embed-text"


def test_switch_rejects_unknown_provider_with_400(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = _switch(harness, provider="cohere")

    assert response.status_code == 400
    assert "Unknown embedding provider 'cohere'" in response.json()["detail"]


@pytest.mark.parametrize(
    ("provider", "missing"),
    [("openai", "OPENAI_API_KEY"), ("ibm", "WATSONX_APIKEY, WATSONX_PROJECT_ID")],
)
def test_switch_without_credentials_keeps_current_index(
    harness_factory: HarnessFactory, provider: str, missing: str
) -> None:
    harness = harness_factory(parser=FakeParser([text_node("a")]))
    _ingest(harness)

    response = _switch(harness, provider=provider)

    assert response.status_code == 400
    assert missing in response.json()["detail"]
    health = harness.client.get("/health").json()
    assert (health["indexed_documents"], health["active_embedding_provider"]) == (1, "ollama")


def test_switch_maps_sdk_construction_failure_to_400(
    harness_factory: HarnessFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llama_index.embeddings.ibm as ibm_module

    def _failing(**_: Any) -> None:
        raise PermissionError("IAM token request failed")

    monkeypatch.setattr(ibm_module, "WatsonxEmbeddings", _failing)
    harness = harness_factory(watsonx_apikey="key", watsonx_project_id="proj")

    response = _switch(harness, provider="ibm")

    assert response.status_code == 400
    assert response.json() == {"detail": "Failed to switch embedding provider: PermissionError."}


def test_switch_rejects_unexpected_fields(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    assert _switch(harness, provider="ollama", api_key="leak").status_code == 422


def test_admin_token_guards_switch_when_configured(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(admin_api_key="s3cret")

    missing = _switch(harness, provider="ollama")
    wrong = _switch(harness, {"X-Admin-Token": "nope"}, provider="ollama")
    right = _switch(harness, {"X-Admin-Token": "s3cret"}, provider="ollama")

    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "X-Admin-Token"
    assert wrong.status_code == 401
    assert right.status_code == 200


def test_openapi_lists_provider_enum(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    schema = harness.client.get("/openapi.json").json()

    provider = schema["components"]["schemas"]["EmbeddingSwitchRequest"]["properties"]["provider"]
    assert provider["enum"] == ["ollama", "ibm", "openai"]
