"""Docling-based document parsing (spec §3.1 steps 4-6).

:class:`DoclingDocumentParser` runs ``DoclingReader`` with the JSON export
(so Docling's structural provenance survives) followed by
``DoclingNodeParser``, which chunks along section / table boundaries and
stores ``doc_items`` / ``headings`` on every node.

A single ``DocumentConverter`` is built per parser and reused: Docling loads
its layout and TableFormer models on the first PDF and caches them on the
converter, so constructing one per request (as a naive implementation
would) reloads the models on every upload.

Chunking defaults to Docling's ``HybridChunker`` with the embedding model's
own tokenizer, so no chunk exceeds the embedding context (512 tokens for
``granite-embedding:278m``); Ollama would otherwise truncate the input
silently and the tail of a long section would never be retrievable.

The :class:`DocumentParser` protocol is the dependency-injection seam the API
uses, so unit tests can substitute a parser that needs no models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

from docling_llamaindex_pipeline.errors import DocumentParsingError

if TYPE_CHECKING:
    from pathlib import Path

    from docling_core.transforms.chunker.base import BaseChunker
    from llama_index.core.schema import BaseNode

    from docling_llamaindex_pipeline.config import Settings

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DoclingDocumentParser",
    "DocumentParser",
    "build_chunker",
    "build_document_converter",
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".xml"}
)
"""File extensions accepted by ``POST /api/v1/ingest`` (spec §4.2)."""

_ALLOWED_FORMATS: list[InputFormat] = [
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.PPTX,
    InputFormat.XLSX,
    InputFormat.HTML,
    InputFormat.CSV,
    # Docling sniffs ``.xml`` content to pick a schema-specific backend.
    InputFormat.XML_JATS,
    InputFormat.XML_USPTO,
]
"""Docling formats matching :data:`SUPPORTED_EXTENSIONS`.

Restricting the converter means a file whose content does not match an
accepted format is rejected by Docling even if its extension passed.
"""


class DocumentParser(Protocol):
    """Converts a file on disk into LlamaIndex nodes."""

    def parse(self, path: Path) -> list[BaseNode]:
        """Parse ``path`` into nodes. Blocking; callers run it off the event loop.

        Raises:
            DocumentParsingError: The document could not be converted.
        """
        ...


def _ocr_options(settings: Settings) -> OcrOptions:
    """Pin the OCR engine; Docling's auto mode may pick PaddleOCR-derived models."""
    if settings.docling_ocr_engine == "easyocr":
        return EasyOcrOptions(lang=settings.docling_ocr_lang)
    return TesseractCliOcrOptions(lang=settings.docling_ocr_lang)


def build_document_converter(settings: Settings) -> DocumentConverter:
    """Build a Docling converter configured from ``settings``.

    Args:
        settings: Source of the OCR engine / languages, the table-structure
            toggle, the accelerator device and the (air-gapped) model
            artifacts path.

    Returns:
        A converter accepting only the supported formats.
    """
    pdf_options = PdfPipelineOptions(
        do_ocr=settings.docling_do_ocr,
        ocr_options=_ocr_options(settings),
        do_table_structure=settings.docling_do_table_structure,
        artifacts_path=settings.docling_artifacts_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice(settings.docling_device)),
    )
    return DocumentConverter(
        allowed_formats=_ALLOWED_FORMATS,
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def build_chunker(settings: Settings) -> BaseChunker:
    """Build the chunker ``DoclingNodeParser`` splits documents with.

    ``hybrid`` loads ``settings.chunk_tokenizer`` from the Hugging Face Hub
    (or the local HF cache; set ``HF_HUB_OFFLINE=1`` on air-gapped hosts).

    Args:
        settings: Source of the chunking strategy, tokenizer and token budget.

    Returns:
        A token-bounded ``HybridChunker`` or an unbounded ``HierarchicalChunker``.

    Raises:
        RuntimeError: The tokenizer could not be loaded.
    """
    if settings.chunker == "hierarchical":
        return HierarchicalChunker()
    try:
        tokenizer = HuggingFaceTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            model_name=settings.chunk_tokenizer,
            max_tokens=settings.chunk_max_tokens,
        )
    except Exception as exc:  # transformers raises OSError / ValueError / HF errors.
        msg = (
            f"Could not load chunk tokenizer '{settings.chunk_tokenizer}': {exc}. "
            "Allow access to the Hugging Face Hub once (it is cached afterwards), "
            "or set CHUNKER=hierarchical."
        )
        raise RuntimeError(msg) from exc
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


class DoclingDocumentParser:
    """:class:`DocumentParser` backed by ``DoclingReader`` + ``DoclingNodeParser``.

    Args:
        converter: The Docling converter to reuse across calls.
        chunker: Chunker for ``DoclingNodeParser``; defaults to Docling's
            ``HierarchicalChunker``.
    """

    def __init__(self, converter: DocumentConverter, chunker: BaseChunker | None = None) -> None:
        self._reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            doc_converter=converter,
        )
        self._node_parser = DoclingNodeParser(chunker=chunker or HierarchicalChunker())

    def parse(self, path: Path) -> list[BaseNode]:
        """Convert ``path`` with Docling and chunk it into nodes.

        Args:
            path: The staged upload.

        Returns:
            One node per structural chunk, with Docling's metadata.

        Raises:
            DocumentParsingError: Docling rejected or failed on the document.
        """
        try:
            documents = self._reader.load_data(path)
            return self._node_parser.get_nodes_from_documents(documents)
        except ConversionError as exc:
            msg = f"Docling parsing failed: {exc}"
            raise DocumentParsingError(msg) from exc
        except Exception as exc:  # Docling backends raise arbitrary types on bad input.
            msg = f"Docling parsing failed ({type(exc).__name__})."
            raise DocumentParsingError(msg) from exc
