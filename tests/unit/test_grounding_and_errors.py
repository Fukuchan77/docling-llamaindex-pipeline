"""Bounding-box extraction (spec §3.2 step 7, §5.1) and error classification (§7.1)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from llama_index.core.schema import NodeWithScore

from docling_llamaindex_pipeline.errors import is_model_server_unreachable
from docling_llamaindex_pipeline.grounding import (
    extract_bounding_boxes,
    extract_headings,
    to_grounded_source,
)
from docling_llamaindex_pipeline.schemas import BoundingBox
from tests.support.docling_docs import mid_term_plan_document, nodes_from_document
from tests.support.fakes import text_node


def test_real_docling_metadata_yields_page_and_bbox() -> None:
    first, second = nodes_from_document(mid_term_plan_document())

    assert extract_bounding_boxes(first.metadata) == [
        BoundingBox(page_no=1, l=72.0, t=520.4, r=523.5, b=480.1, coord_origin="BOTTOMLEFT")
    ]
    assert extract_bounding_boxes(second.metadata)[0].page_no == 2
    assert extract_headings(second.metadata) == ["中期経営計画", "施策"]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"doc_items": "not a list"},
        {"doc_items": [None, 1, "x"]},
        {"doc_items": [{"prov": "nope"}]},
        {"doc_items": [{"prov": [None, {"page_no": "first"}, {"page_no": 1}]}]},
    ],
)
def test_malformed_metadata_is_skipped(metadata: dict[str, Any]) -> None:
    assert extract_bounding_boxes(metadata) == []


def test_missing_coord_origin_defaults_to_bottomleft() -> None:
    metadata = {"doc_items": [{"prov": [{"page_no": 3, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]}]}

    [box] = extract_bounding_boxes(metadata)

    assert (box.page_no, box.coord_origin) == (3, "BOTTOMLEFT")


def test_headings_ignore_non_strings() -> None:
    assert extract_headings({"headings": ["a", 1, None, "b"]}) == ["a", "b"]
    assert extract_headings({"headings": "a"}) == []


def test_grounded_source_truncates_snippet_and_drops_bad_file_name() -> None:
    node = text_node("x" * 500, file_name=42)

    source = to_grounded_source(NodeWithScore(node=node, score=0.5), export_grounding=True)

    assert len(source.text_snippet) == 300
    assert source.file_name is None
    assert source.score == 0.5


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://model")


class _SdkError(Exception):
    pass


def _wrapped(inner: BaseException) -> BaseException:
    try:
        raise _SdkError("wrapper") from inner
    except _SdkError as exc:
        return exc


def _implicitly_chained(inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException:  # noqa: BLE001
            raise _SdkError("during handling") from None
    except _SdkError as exc:
        exc.__suppress_context__ = False
        exc.__context__ = inner
        return exc


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused", request=_request()),
        httpx.ReadTimeout("slow", request=_request()),
        ConnectionError("Failed to connect to Ollama"),
        TimeoutError(),
        _wrapped(httpx.ConnectError("refused", request=_request())),
        _implicitly_chained(ConnectionRefusedError()),
    ],
)
def test_network_failures_are_detected(exc: BaseException) -> None:
    assert is_model_server_unreachable(exc)


@pytest.mark.parametrize("exc", [ValueError("bad"), _wrapped(KeyError("k"))])
def test_other_failures_are_not_network(exc: BaseException) -> None:
    assert not is_model_server_unreachable(exc)


def test_cyclic_cause_chain_terminates() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a

    assert not is_model_server_unreachable(a)
