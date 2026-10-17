"""Live end-to-end lane: real Docling, real Ollama embeddings and LLM.

Skipped unless ``RUN_INTEGRATION_OLLAMA=1``. Needs a reachable Ollama daemon
(``OLLAMA_BASE_URL``) with ``OLLAMA_EMBED_MODEL`` and ``OLLAMA_LLM_MODEL``
pulled. The PDF case also needs Docling's layout models: either network
access to the Hugging Face Hub on first run, or ``DOCLING_ARTIFACTS_PATH``
pointing at a pre-downloaded copy. Run it with ``mise run test:integration``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from docling_llamaindex_pipeline.app import create_app
from docling_llamaindex_pipeline.config import Settings
from tests.support.pdf import minimal_pdf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION_OLLAMA") != "1",
        reason="set RUN_INTEGRATION_OLLAMA=1 to run against a live Ollama daemon",
    ),
]

HTML = b"""<html><body>
<h1>Mid-term plan</h1><h2>Financial targets</h2>
<p>The operating margin target for fiscal 2025 is 12.5 percent.</p>
<p>Key measures are supply-chain automation and migration to the cloud.</p>
</body></html>"""


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Unit runs force the HF Hub offline; this lane may need to fetch models.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    settings = Settings(temp_upload_dir=tmp_path)  # Reads OLLAMA_* etc. from the environment.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_html_round_trip(client: TestClient) -> None:
    ingest = client.post("/api/v1/ingest", files={"file": ("plan.html", HTML)})
    assert ingest.status_code == 201, ingest.text
    assert ingest.json()["total_nodes_created"] >= 1

    query = client.post(
        "/api/v1/query", json={"query": "What is the operating margin target for 2025?", "top_k": 2}
    )

    assert query.status_code == 200, query.text
    body = query.json()
    assert body["answer"].strip()
    assert body["sources"][0]["file_name"] == "plan.html"
    assert "Financial targets" in body["sources"][0]["headings"]


def test_pdf_sources_carry_bounding_boxes(client: TestClient) -> None:
    pdf = minimal_pdf(
        [
            "Operating margin target for fiscal 2025: 12.5 percent.",
            "Measures: supply-chain automation and cloud migration.",
        ]
    )
    ingest = client.post("/api/v1/ingest", files={"file": ("plan.pdf", pdf)})
    assert ingest.status_code == 201, ingest.text

    query = client.post("/api/v1/query", json={"query": "operating margin target", "top_k": 1})

    assert query.status_code == 200, query.text
    [source] = query.json()["sources"]
    [box, *_] = source["bounding_boxes"]
    assert box["page_no"] == 1
    assert box["coord_origin"] == "BOTTOMLEFT"
    assert 0 <= box["l"] < box["r"] <= 612
    assert 0 <= box["b"] < box["t"] <= 792
