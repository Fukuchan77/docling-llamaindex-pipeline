"""``POST /api/v1/ingest`` (spec §3.1, §4.2, §6.3.2, §7.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docling_llamaindex_pipeline.errors import DocumentParsingError
from tests.support.docling_docs import mid_term_plan_document, nodes_from_document
from tests.support.fakes import BrokenEmbedding, FakeParser, UnreachableEmbedding, text_node

if TYPE_CHECKING:
    import httpx2

    from tests.conftest import Harness, HarnessFactory

PDF_BYTES = b"%PDF-1.7 fake"


def _upload(harness: Harness, name: str, content: bytes = PDF_BYTES) -> httpx2.Response:
    return harness.client.post("/api/v1/ingest", files={"file": (name, content)})


def _staging_is_clean(harness: Harness) -> bool:
    return not harness.upload_dir.exists() or not any(harness.upload_dir.iterdir())


def test_ingest_indexes_nodes_and_returns_preview(harness_factory: HarnessFactory) -> None:
    parser = FakeParser(factory=lambda: nodes_from_document(mid_term_plan_document()))
    harness = harness_factory(parser=parser)

    response = _upload(harness, "mid_term_plan.pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["file_name"] == "mid_term_plan.pdf"
    assert body["total_nodes_created"] == 2
    assert body["active_embedding_provider"] == "ollama"
    assert len(body["sample_nodes"]) == 2
    assert body["sample_nodes"][0]["headings"] == ["中期経営計画", "重点財務目標"]
    assert "12.5%" in body["sample_nodes"][0]["text"]

    stats = harness.container.rag_service.stats
    assert (stats.documents, stats.nodes) == (1, 2)


def test_ingest_stages_upload_under_original_name_and_deletes_it(
    harness_factory: HarnessFactory,
) -> None:
    harness = harness_factory(parser=FakeParser([text_node("hello")]))

    response = _upload(harness, "report.pdf", b"%PDF payload")

    assert response.status_code == 201
    [staged_path] = harness.parser.calls
    assert staged_path.name == "report.pdf"
    assert staged_path.parent.parent == harness.upload_dir
    assert harness.parser.file_existed == [True]
    assert harness.parser.contents == [b"%PDF payload"]
    assert not staged_path.exists()
    assert _staging_is_clean(harness)


def test_ingest_tags_nodes_with_file_name(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([text_node("営業利益率は12.5%")]))
    _upload(harness, "plan.pdf")

    result = harness.client.post("/api/v1/query", json={"query": "営業利益率"})

    assert result.json()["sources"][0]["file_name"] == "plan.pdf"


def test_ingest_strips_directories_from_client_file_name(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([text_node("x")]))

    response = _upload(harness, "../../etc/evil.pdf")

    assert response.status_code == 201
    assert response.json()["file_name"] == "evil.pdf"
    assert harness.parser.calls[0].parent.parent == harness.upload_dir


@pytest.mark.parametrize("name", ["notes.txt", "image.png", "archive.pdf.exe", "no_extension"])
def test_ingest_rejects_unsupported_extension(harness_factory: HarnessFactory, name: str) -> None:
    harness = harness_factory()

    response = _upload(harness, name)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unsupported format")
    assert harness.parser.calls == []
    assert _staging_is_clean(harness)


@pytest.mark.parametrize(
    "name", ["a.PDF", "b.docx", "c.pptx", "d.xlsx", "e.html", "f.csv", "g.xml"]
)
def test_ingest_accepts_every_supported_extension(
    harness_factory: HarnessFactory, name: str
) -> None:
    harness = harness_factory(parser=FakeParser([text_node("content")]))

    assert _upload(harness, name).status_code == 201


def test_ingest_rejects_empty_file(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    response = _upload(harness, "empty.pdf", b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert harness.parser.calls == []
    assert _staging_is_clean(harness)


def test_ingest_rejects_file_over_limit(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(max_upload_size_mb=1)

    # Under the Content-Length guard (limit + 1 MiB multipart allowance), over the file limit.
    response = _upload(harness, "big.pdf", b"x" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert harness.parser.calls == []
    assert _staging_is_clean(harness)


def test_ingest_rejects_oversized_body_before_parsing_it(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(max_upload_size_mb=1)

    response = _upload(harness, "huge.pdf", b"x" * (3 * 1024 * 1024))

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the upload limit."}


def test_ingest_maps_parse_failure_to_500_and_cleans_up(harness_factory: HarnessFactory) -> None:
    parser = FakeParser(error=DocumentParsingError("Docling parsing failed: encrypted PDF"))
    harness = harness_factory(parser=parser)

    response = _upload(harness, "locked.pdf")

    assert response.status_code == 500
    assert response.json() == {"detail": "Docling parsing failed: encrypted PDF"}
    assert harness.container.rag_service.stats.documents == 0
    assert _staging_is_clean(harness)


def test_ingest_rejects_document_without_content(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([]))

    response = _upload(harness, "scan.pdf")

    assert response.status_code == 400
    assert "No extractable content" in response.json()["detail"]
    assert harness.container.rag_service.stats.documents == 0


def test_ingest_maps_unreachable_embedding_server_to_503(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(
        parser=FakeParser([text_node("x")]), embed_model=UnreachableEmbedding()
    )

    response = _upload(harness, "doc.pdf")

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert harness.container.rag_service.stats.documents == 0
    assert _staging_is_clean(harness)


def test_ingest_maps_other_embedding_failure_to_500(harness_factory: HarnessFactory) -> None:
    harness = harness_factory(parser=FakeParser([text_node("x")]), embed_model=BrokenEmbedding())

    response = _upload(harness, "doc.pdf")

    assert response.status_code == 500
    assert response.json() == {"detail": "Indexing failed while embedding the document."}


def test_ingest_requires_a_file(harness_factory: HarnessFactory) -> None:
    harness = harness_factory()

    assert harness.client.post("/api/v1/ingest").status_code == 422
