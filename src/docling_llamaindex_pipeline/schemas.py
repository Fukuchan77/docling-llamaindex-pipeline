"""Wire contract of the REST API (spec §4, §5.2).

Field declarations and documentation only; behaviour lives elsewhere.
Request bodies use ``extra="forbid"`` so unexpected client fields are
rejected with HTTP 422 instead of being silently ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from docling_llamaindex_pipeline.config import EmbeddingProvider

__all__ = [
    "BoundingBox",
    "EmbeddingSwitchRequest",
    "EmbeddingSwitchResponse",
    "GroundedSourceNode",
    "HealthResponse",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SampleNode",
]

QUERY_MAX_LENGTH = 8192
"""Upper bound on ``QueryRequest.query``; rejects body-amplification abuse with 422."""


class BoundingBox(BaseModel):
    """A region of the source page a chunk was extracted from (spec §5.1).

    Docling reports PDF coordinates in points with a bottom-left origin. To
    draw on a top-left canvas: ``top = page_height - t``, ``left = l``,
    ``width = r - l``, ``height = t - b``.
    """

    page_no: int | None = Field(default=None, description="1-indexed page number.")
    l: float | None = Field(default=None, description="Left edge X coordinate.")  # noqa: E741 — spec field name.
    t: float | None = Field(default=None, description="Top edge Y coordinate.")
    r: float | None = Field(default=None, description="Right edge X coordinate.")
    b: float | None = Field(default=None, description="Bottom edge Y coordinate.")
    coord_origin: str | None = Field(default="BOTTOMLEFT", description="Coordinate origin.")


class GroundedSourceNode(BaseModel):
    """A retrieved chunk together with where it came from."""

    node_id: str
    text_snippet: str
    score: float | None = None
    headings: list[str] = Field(default_factory=list[str])
    bounding_boxes: list[BoundingBox] = Field(default_factory=list[BoundingBox])
    file_name: str | None = None


class SampleNode(BaseModel):
    """Preview of one node created by an ingest call."""

    node_id: str
    text: str
    headings: list[str] = Field(default_factory=list[str])


class IngestResponse(BaseModel):
    """Body of a successful ``POST /api/v1/ingest``."""

    status: Literal["SUCCESS"] = "SUCCESS"
    file_name: str
    total_nodes_created: int
    active_embedding_provider: EmbeddingProvider
    sample_nodes: list[SampleNode]


class QueryRequest(BaseModel):
    """Body of ``POST /api/v1/query``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
        description="User question.",
    )
    top_k: int = Field(default=4, ge=1, le=20, description="Number of source nodes to retrieve.")
    export_grounding: bool = Field(
        default=True,
        description="Whether to include bounding-box coordinates in each source.",
    )


class QueryResponse(BaseModel):
    """Body of a successful ``POST /api/v1/query``."""

    query: str
    answer: str
    provider: EmbeddingProvider
    sources: list[GroundedSourceNode]


class EmbeddingSwitchRequest(BaseModel):
    """Body of ``POST /api/v1/config/embedding``.

    ``provider`` is a plain string rather than :class:`EmbeddingProvider` so
    an unknown value is answered with the spec's HTTP 400 instead of
    FastAPI's generic 422; the allowed values are still published in the
    OpenAPI schema.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        description="Embedding provider to activate.",
        json_schema_extra={"enum": [p.value for p in EmbeddingProvider]},
    )
    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Overrides the provider's default embedding model.",
    )


class EmbeddingSwitchResponse(BaseModel):
    """Body of a successful ``POST /api/v1/config/embedding``."""

    status: Literal["UPDATED"] = "UPDATED"
    active_provider: EmbeddingProvider
    embedding_model: str
    message: str


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["healthy"] = "healthy"
    indexed_documents: int
    indexed_nodes: int
    active_embedding_provider: EmbeddingProvider
    embedding_model: str
