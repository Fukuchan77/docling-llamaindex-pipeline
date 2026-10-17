"""``GET /health`` (spec §7.2).

Liveness / readiness probe target. It reports in-memory state only and never
calls a model server, so the API stays "up" while Ollama restarts.
"""

from __future__ import annotations

from fastapi import APIRouter

from docling_llamaindex_pipeline.api.deps import ContainerDep
from docling_llamaindex_pipeline.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
def health_check(container: ContainerDep) -> HealthResponse:
    """Return index counts and the active embedding provider."""
    stats = container.rag_service.stats
    return HealthResponse(
        indexed_documents=stats.documents,
        indexed_nodes=stats.nodes,
        active_embedding_provider=stats.embedding_provider,
        embedding_model=stats.embedding_model,
    )
