"""``POST /api/v1/query`` (spec §3.2, §4.3, §7.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.support.docling_docs import mid_term_plan_document, nodes_from_document
from tests.support.fakes import FakeParser, FixedLLM, text_node, unreachable_llm

if TYPE_CHECKING:
    from tests.conftest import Harness, HarnessFactory

QUESTION = "2025年度の営業利益率の目標値とその根拠となる施策を教えてください。"


def _ingest_plan(harness: Harness) -> None:
    response = harness.client.post("/api/v1/ingest", files={"file": ("mid_term_plan.pdf", b"%PDF")})
    assert response.status_code == 201


def _plan_harness(harness_factory: HarnessFactory, **kwargs: Any) -> Harness:
    parser = FakeParser(factory=lambda: nodes_from_document(mid_term_plan_document()))
    harness = harness_factory(parser=parser, **kwargs)
    _ingest_plan(harness)
    return harness


def test_query_before_ingest_is_rejected(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = harness.client.post("/api/v1/query", json={"query": QUESTION})

    assert response.status_code == 400
    assert "No documents have been indexed" in response.json()["detail"]


def test_query_returns_answer_with_grounded_sources(harness_factory: HarnessFactory) -> None:
    llm = FixedLLM(answer="2025年度の営業利益率目標は12.5%です。")
    harness = _plan_harness(harness_factory, llm=llm)

    response = harness.client.post(
        "/api/v1/query", json={"query": "営業利益率 12.5% 目標", "top_k": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "営業利益率 12.5% 目標"
    assert body["answer"] == "2025年度の営業利益率目標は12.5%です。"
    assert body["provider"] == "ollama"
    [source] = body["sources"]
    assert "12.5%" in source["text_snippet"]
    assert source["headings"] == ["中期経営計画", "重点財務目標"]
    assert source["file_name"] == "mid_term_plan.pdf"
    assert isinstance(source["score"], float)
    assert source["bounding_boxes"] == [
        {
            "page_no": 1,
            "l": 72.0,
            "t": 520.4,
            "r": 523.5,
            "b": 480.1,
            "coord_origin": "BOTTOMLEFT",
        }
    ]
    # The retrieved chunk text reaches the LLM as context.
    assert any("12.5%" in prompt for prompt in llm.prompts)


def test_query_respects_top_k(harness_factory: HarnessFactory) -> None:
    harness = _plan_harness(harness_factory)

    one = harness.client.post("/api/v1/query", json={"query": QUESTION, "top_k": 1})
    two = harness.client.post("/api/v1/query", json={"query": QUESTION, "top_k": 2})

    assert len(one.json()["sources"]) == 1
    assert len(two.json()["sources"]) == 2


def test_query_defaults_top_k_to_four(harness_factory: HarnessFactory) -> None:
    nodes = [text_node(f"chunk number {i}") for i in range(6)]
    harness = harness_factory(parser=FakeParser(nodes))
    _ingest_plan(harness)

    response = harness.client.post("/api/v1/query", json={"query": "chunk number"})

    assert len(response.json()["sources"]) == 4


def test_query_without_grounding_omits_bounding_boxes(harness_factory: HarnessFactory) -> None:
    harness = _plan_harness(harness_factory)

    response = harness.client.post(
        "/api/v1/query", json={"query": QUESTION, "export_grounding": False}
    )

    sources = response.json()["sources"]
    assert sources
    assert all(source["bounding_boxes"] == [] for source in sources)
    assert all(source["headings"] for source in sources)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "x" * 8193},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": 21},
        {"query": "q", "unexpected": True},
    ],
)
def test_query_validates_request_body(
    harness_factory: HarnessFactory, body: dict[str, Any]
) -> None:
    harness = harness_factory()

    assert harness.client.post("/api/v1/query", json=body).status_code == 422


def test_query_maps_unreachable_llm_to_503(harness_factory: HarnessFactory) -> None:
    harness = _plan_harness(harness_factory, llm=unreachable_llm())

    response = harness.client.post("/api/v1/query", json={"query": QUESTION})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model server unreachable."}


def test_query_maps_other_failures_to_500(harness_factory: HarnessFactory) -> None:
    harness = _plan_harness(harness_factory, llm=FixedLLM(failure=RuntimeError("boom")))

    response = harness.client.post("/api/v1/query", json={"query": QUESTION})

    assert response.status_code == 500
    assert response.json() == {"detail": "Query execution failed."}


def test_query_times_out_with_504(harness_factory: HarnessFactory) -> None:
    harness = _plan_harness(harness_factory, llm=FixedLLM(delay=1.0), query_timeout_seconds=0.05)

    response = harness.client.post("/api/v1/query", json={"query": QUESTION})

    assert response.status_code == 504
    assert response.json() == {"detail": "Query exceeded the 0.05s timeout."}
