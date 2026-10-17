"""Source grounding: turn node metadata into citations (spec §3.2 step 7).

``DoclingNodeParser`` stores Docling's chunk metadata on every node:
``headings`` (the section path) and ``doc_items`` (the document elements the
chunk was built from), each carrying ``prov`` entries with ``page_no`` and a
``bbox``. Formats without a page layout (HTML, DOCX, CSV, ...) yield empty
``prov`` lists, so their sources simply carry no bounding boxes.

Metadata is untrusted structure (it round-trips through the vector store),
so it is validated element by element: a malformed entry is skipped rather
than failing the whole query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from docling_llamaindex_pipeline.schemas import BoundingBox, GroundedSourceNode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from llama_index.core.schema import NodeWithScore

__all__ = ["extract_bounding_boxes", "extract_headings", "to_grounded_source"]

SNIPPET_LENGTH = 300
"""Characters of node text returned as ``text_snippet``."""


class _Box(BaseModel):
    model_config = ConfigDict(extra="ignore")

    l: float | None = None  # noqa: E741 — Docling field name.
    t: float | None = None
    r: float | None = None
    b: float | None = None
    coord_origin: str | None = None


class _Prov(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_no: int | None = None
    bbox: _Box | None = None


class _DocItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prov: list[Any] = []


def extract_bounding_boxes(metadata: Mapping[str, Any]) -> list[BoundingBox]:
    """Collect every provenance bounding box from a node's ``doc_items``.

    Args:
        metadata: The node's metadata as written by ``DoclingNodeParser``.

    Returns:
        Bounding boxes in document order. Entries without a ``bbox`` are
        skipped; malformed entries are ignored.
    """
    raw_items = metadata.get("doc_items")
    if not isinstance(raw_items, list):
        return []

    boxes: list[BoundingBox] = []
    for raw_item in raw_items:  # pyright: ignore[reportUnknownVariableType]
        try:
            item = _DocItem.model_validate(raw_item)
        except ValidationError:
            continue
        for raw_prov in item.prov:
            try:
                prov = _Prov.model_validate(raw_prov)
            except ValidationError:
                continue
            if prov.bbox is None:
                continue
            boxes.append(
                BoundingBox(
                    page_no=prov.page_no,
                    l=prov.bbox.l,
                    t=prov.bbox.t,
                    r=prov.bbox.r,
                    b=prov.bbox.b,
                    coord_origin=prov.bbox.coord_origin or "BOTTOMLEFT",
                )
            )
    return boxes


def extract_headings(metadata: Mapping[str, Any]) -> list[str]:
    """Return the chunk's section path, or ``[]`` when absent or malformed."""
    headings = metadata.get("headings")
    if not isinstance(headings, list):
        return []
    return [h for h in headings if isinstance(h, str)]  # pyright: ignore[reportUnknownVariableType]


def to_grounded_source(source: NodeWithScore, *, export_grounding: bool) -> GroundedSourceNode:
    """Convert a retrieved node into its API representation.

    Args:
        source: A node returned by the retriever, with its similarity score.
        export_grounding: Whether to include bounding boxes.

    Returns:
        The citation for ``source``.
    """
    node = source.node
    metadata = node.metadata
    file_name = metadata.get("file_name")
    return GroundedSourceNode(
        node_id=node.node_id,
        text_snippet=node.get_content()[:SNIPPET_LENGTH].strip(),
        score=source.score,
        headings=extract_headings(metadata),
        bounding_boxes=extract_bounding_boxes(metadata) if export_grounding else [],
        file_name=file_name if isinstance(file_name, str) else None,
    )
