"""Build real Docling chunk metadata without running any layout model.

A PDF's provenance (page numbers, bounding boxes) normally comes from
Docling's layout model, which needs downloaded weights. Constructing the
``DoclingDocument`` directly and chunking it with the real
``DoclingNodeParser`` yields node metadata byte-identical in shape to a
converted PDF, fully offline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from llama_index.core import Document
from llama_index.node_parser.docling import DoclingNodeParser

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

PLAN_TEXT = "中期経営計画目標: 2025年度営業利益率 12.5% を必達目標とする。"
SUPPLY_TEXT = "主な施策はサプライチェーンの自動化とクラウド移行によるコスト削減である。"


def _prov(page_no: int, l: float, t: float, r: float, b: float, length: int) -> ProvenanceItem:  # noqa: E741
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, length),
    )


def mid_term_plan_document() -> DoclingDocument:
    """A two-page "PDF" with headings and provenance on every item."""
    doc = DoclingDocument(name="mid_term_plan")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=612, height=792))
    doc.add_heading(text="中期経営計画", prov=_prov(1, 72, 760, 300, 740, 6))
    doc.add_heading(text="重点財務目標", level=2, prov=_prov(1, 72, 700, 300, 680, 6))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=PLAN_TEXT,
        prov=_prov(1, 72.0, 520.4, 523.5, 480.1, len(PLAN_TEXT)),
    )
    doc.add_heading(text="施策", level=2, prov=_prov(2, 72, 760, 300, 740, 2))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=SUPPLY_TEXT,
        prov=_prov(2, 72.0, 700.0, 540.0, 640.0, len(SUPPLY_TEXT)),
    )
    return doc


def nodes_from_document(doc: DoclingDocument) -> list[BaseNode]:
    """Chunk ``doc`` exactly as the production parser does after ``DoclingReader``."""
    li_doc = Document(text=json.dumps(doc.export_to_dict()))
    return DoclingNodeParser().get_nodes_from_documents([li_doc])
