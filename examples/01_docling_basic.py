"""Convert a document with Docling: standard pipeline and Granite-Docling VLM.

Usage::

    uv run python examples/01_docling_basic.py path/to/file.pdf
    uv run python examples/01_docling_basic.py path/to/file.pdf --vlm

Standard pipeline: IBM's layout model + TableFormer, OCR pinned to Tesseract
(Docling's automatic OCR selection may otherwise pick PaddleOCR-derived models).
``--vlm``: the IBM Granite-Docling-258M vision-language model, pinned
explicitly rather than relying on Docling's default VLM preset.

Docs: https://docling-project.github.io/docling/usage/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def standard_converter() -> DocumentConverter:
    """Layout analysis + TableFormer, Tesseract OCR for Japanese and English."""
    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=TesseractCliOcrOptions(lang=["iso:jpn", "iso:eng"]),
        do_table_structure=True,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def granite_docling_converter() -> DocumentConverter:
    """Page images -> DocTags with Granite-Docling-258M (Hugging Face transformers)."""
    options = VlmPipelineOptions(vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=options)
        }
    )


def main() -> None:
    """Convert the given file and write Markdown and JSON next to ``output/``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("source", help="Local path or URL of the document")
    parser.add_argument("--vlm", action="store_true", help="Use Granite-Docling-258M")
    args = parser.parse_args()

    converter = granite_docling_converter() if args.vlm else standard_converter()
    document = converter.convert(args.source).document

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stem = Path(str(args.source)).stem
    document.save_as_markdown(out_dir / f"{stem}.md")
    document.save_as_json(out_dir / f"{stem}.json")
    print(document.export_to_markdown()[:400])
    print(f"\nWrote {out_dir / stem}.md / .json")


if __name__ == "__main__":
    main()
