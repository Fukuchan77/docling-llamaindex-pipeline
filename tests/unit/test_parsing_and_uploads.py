"""Real Docling parsing for model-free formats, and upload sanitising (spec §3.1, §6.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from docling.datamodel.base_models import InputFormat
from docling.exceptions import ConversionError

from docling_llamaindex_pipeline.errors import DocumentParsingError, UnsupportedFormatError
from docling_llamaindex_pipeline.parsing import (
    SUPPORTED_EXTENSIONS,
    DoclingDocumentParser,
    build_document_converter,
)
from docling_llamaindex_pipeline.uploads import safe_filename, validate_extension

if TYPE_CHECKING:
    from pathlib import Path

    from docling.document_converter import DocumentConverter

    from tests.conftest import SettingsFactory

HTML = """<html><body>
<h1>財務ハイライト</h1><h2>連結損益計算書</h2>
<p>当第3四半期累計期間における売上高は前年同期比10%増加した。</p>
<table><tr><th>年度</th><th>営業利益率</th></tr><tr><td>2025</td><td>12.5%</td></tr></table>
</body></html>"""


@pytest.fixture(scope="module")
def converter() -> DocumentConverter:
    from docling_llamaindex_pipeline.config import Settings

    return build_document_converter(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]


def test_html_is_chunked_by_structure_with_headings(
    converter: DocumentConverter, tmp_path: Path
) -> None:
    path = tmp_path / "report.html"
    path.write_text(HTML, encoding="utf-8")

    paragraph, table = DoclingDocumentParser(converter).parse(path)

    assert "売上高" in paragraph.get_content()
    assert paragraph.metadata["headings"] == ["財務ハイライト", "連結損益計算書"]
    assert table.metadata["doc_items"][0]["label"] == "table"
    assert "12.5%" in table.get_content()
    # Provenance stays out of the embedded text (DoclingNodeParser exclusions).
    assert "doc_items" in table.excluded_embed_metadata_keys


def test_csv_table_is_preserved(converter: DocumentConverter, tmp_path: Path) -> None:
    path = tmp_path / "margins.csv"
    path.write_text("year,margin\n2025,12.5\n2026,13.0\n", encoding="utf-8")

    [node] = DoclingDocumentParser(converter).parse(path)

    assert "2025" in node.get_content()
    assert "13.0" in node.get_content()


def test_corrupt_document_raises_parsing_error(
    converter: DocumentConverter, tmp_path: Path
) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentParsingError, match="Docling parsing failed"):
        DoclingDocumentParser(converter).parse(path)


def test_conversion_error_message_is_kept(
    converter: DocumentConverter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = DoclingDocumentParser(converter)

    def _raise(*_: object, **__: object) -> None:
        raise ConversionError("Input document x.pdf is not valid.")

    monkeypatch.setattr(converter, "convert", _raise)

    with pytest.raises(DocumentParsingError, match=r"Docling parsing failed: Input document"):
        parser.parse(tmp_path / "x.pdf")


def test_unexpected_backend_error_is_wrapped(
    converter: DocumentConverter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = DoclingDocumentParser(converter)

    def _raise(*_: object, **__: object) -> None:
        raise KeyError("/word/document.xml")

    monkeypatch.setattr(converter, "convert", _raise)

    with pytest.raises(DocumentParsingError, match=r"^Docling parsing failed \(KeyError\)\.$"):
        parser.parse(tmp_path / "x.docx")


def test_converter_respects_settings(settings_factory: SettingsFactory, tmp_path: Path) -> None:
    settings = settings_factory(
        docling_do_ocr=False,
        docling_do_table_structure=False,
        docling_device="cpu",
        docling_artifacts_path=tmp_path,
    )

    converter = build_document_converter(settings)

    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options is not None
    assert options.artifacts_path == tmp_path
    assert getattr(options, "do_ocr") is False  # noqa: B009 — PdfPipelineOptions field.
    assert getattr(options, "do_table_structure") is False  # noqa: B009
    assert set(converter.allowed_formats) >= {InputFormat.PDF, InputFormat.CSV}
    assert InputFormat.IMAGE not in converter.allowed_formats


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\evil.docx", "evil.docx"),
        ("/abs/path/file.csv", "file.csv"),
        ("a\x00b<c>.html", "a_b_c_.html"),
        (".hidden.pdf", "hidden.pdf"),
        ("", "upload"),
        (None, "upload"),
        ("..", "upload"),
        ("x" * 300 + ".pdf", "x" * 196 + ".pdf"),
    ],
)
def test_safe_filename(raw: str | None, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_validate_extension_is_case_insensitive() -> None:
    assert validate_extension("A.PDF") == ".pdf"
    assert {validate_extension(f"x{ext}") for ext in SUPPORTED_EXTENSIONS} == SUPPORTED_EXTENSIONS
    with pytest.raises(UnsupportedFormatError, match=r"'\(none\)'"):
        validate_extension("README")


def test_minimal_pdf_fixture_is_a_readable_pdf() -> None:
    """Guards the integration lane's PDF generator (pypdfium2 ships with Docling)."""
    import pypdfium2

    from tests.support.pdf import minimal_pdf

    document = pypdfium2.PdfDocument(minimal_pdf(["Margin (2025): 12.5 percent", "Second line"]))

    assert len(document) == 1
    text = document[0].get_textpage().get_text_range()
    assert "Margin (2025): 12.5 percent" in text
    assert "Second line" in text
