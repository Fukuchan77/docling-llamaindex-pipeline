"""``POST /api/v1/ingest`` (spec §3.1, §4.2).

Validate → stage to a private temp dir → parse with Docling (in a worker
thread, bounded by ``MAX_CONCURRENT_INGESTIONS``) → tag nodes with the file
name → embed and insert → delete the staged file → respond.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, UploadFile, status

from docling_llamaindex_pipeline.api.deps import ContainerDep
from docling_llamaindex_pipeline.errors import NoExtractableContentError
from docling_llamaindex_pipeline.grounding import extract_headings
from docling_llamaindex_pipeline.schemas import IngestResponse, SampleNode
from docling_llamaindex_pipeline.uploads import stage_upload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.core.schema import BaseNode

router = APIRouter(prefix="/api/v1", tags=["ingest"])
logger = logging.getLogger(__name__)

PREVIEW_NODES = 2
PREVIEW_LENGTH = 160


def _preview(nodes: Sequence[BaseNode]) -> list[SampleNode]:
    return [
        SampleNode(
            node_id=node.node_id,
            text=node.get_content()[:PREVIEW_LENGTH].replace("\n", " "),
            headings=extract_headings(node.metadata),
        )
        for node in nodes[:PREVIEW_NODES]
    ]


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    summary="Parse and index a document via Docling",
    responses={
        400: {"description": "Unsupported format, empty file, or no extractable content."},
        413: {"description": "File exceeds MAX_UPLOAD_SIZE_MB."},
        500: {"description": "Docling parsing or indexing failed."},
        503: {"description": "Embedding server unreachable."},
    },
)
async def ingest_document(
    container: ContainerDep,
    file: UploadFile = File(  # noqa: B008 — FastAPI parameter declaration idiom.
        description="PDF, DOCX, PPTX, XLSX, HTML, CSV or XML file.",
    ),
) -> IngestResponse:
    """Parse ``file`` with Docling and index its chunks with provenance."""
    settings = container.settings
    async with (
        stage_upload(
            file,
            upload_dir=settings.temp_upload_dir,
            max_bytes=settings.max_upload_bytes,
        ) as staged,
        container.parse_slots,
    ):
        nodes = await asyncio.to_thread(container.parser.parse, staged.path)

    if not nodes:
        msg = f"No extractable content found in '{staged.file_name}'."
        raise NoExtractableContentError(msg)
    for node in nodes:
        node.metadata["file_name"] = staged.file_name

    provider = await container.rag_service.add_document(nodes)
    logger.info(
        "Ingested %s (%d bytes) into %d nodes.", staged.file_name, staged.size_bytes, len(nodes)
    )
    return IngestResponse(
        file_name=staged.file_name,
        total_nodes_created=len(nodes),
        active_embedding_provider=provider,
        sample_nodes=_preview(nodes),
    )
