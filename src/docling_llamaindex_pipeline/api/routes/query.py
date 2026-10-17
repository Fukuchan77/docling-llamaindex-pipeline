"""``POST /api/v1/query`` (spec §3.2, §4.3)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from docling_llamaindex_pipeline.api.deps import ContainerDep
from docling_llamaindex_pipeline.errors import QueryTimeoutError
from docling_llamaindex_pipeline.grounding import to_grounded_source
from docling_llamaindex_pipeline.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    summary="Query the index with source grounding",
    responses={
        400: {"description": "No documents indexed yet."},
        503: {"description": "Embedding or LLM server unreachable."},
        504: {"description": "Query exceeded QUERY_TIMEOUT_SECONDS."},
    },
)
async def query_rag(request: QueryRequest, container: ContainerDep) -> QueryResponse:
    """Answer ``request.query`` and cite each source with page / bounding-box provenance."""
    timeout = container.settings.query_timeout_seconds
    try:
        async with asyncio.timeout(timeout):
            result = await container.rag_service.query(request.query, top_k=request.top_k)
    except TimeoutError as exc:
        msg = f"Query exceeded the {timeout:g}s timeout."
        raise QueryTimeoutError(msg) from exc

    return QueryResponse(
        query=request.query,
        answer=result.answer,
        provider=result.embedding_provider,
        sources=[
            to_grounded_source(source, export_grounding=request.export_grounding)
            for source in result.sources
        ],
    )
