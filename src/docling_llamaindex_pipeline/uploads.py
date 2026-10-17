"""Upload validation and temporary staging (spec §3.1 steps 2-3 and 9, §6.3.2).

Each upload is written to its own ``mkdtemp`` directory (mode ``0700``) under
``TEMP_UPLOAD_DIR`` using a sanitised copy of the client's file name, and the
directory is removed in a ``finally`` block once parsing is done, success or
not. Keeping the original name (rather than a random one) matters because
Docling records it as the document's origin.

The client-supplied name is never used as a path: :func:`safe_filename`
strips directory components, so ``../../etc/passwd`` cannot escape the
staging directory.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from docling_llamaindex_pipeline.errors import (
    EmptyUploadError,
    UnsupportedFormatError,
    UploadTooLargeError,
)
from docling_llamaindex_pipeline.parsing import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import UploadFile

__all__ = ["StagedUpload", "safe_filename", "stage_upload", "validate_extension"]

_CHUNK_SIZE = 1024 * 1024
_MAX_NAME_LENGTH = 200
_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*]")


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """An upload written to disk for parsing.

    Attributes:
        path: Location of the staged file.
        file_name: Sanitised client file name.
        size_bytes: Number of bytes written.
    """

    path: Path
    file_name: str
    size_bytes: int


def safe_filename(raw: str | None) -> str:
    """Reduce a client-supplied file name to a safe single path component.

    Both slash and backslash separators are stripped, control and reserved
    characters are replaced, and the stem is truncated while preserving the
    extension.

    Args:
        raw: The multipart ``filename``; may be ``None`` or empty.

    Returns:
        A non-empty name with no directory part. ``"upload"`` when nothing
        usable remains.
    """
    name = unicodedata.normalize("NFC", raw or "")
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = _UNSAFE_CHARS.sub("_", name).strip().lstrip(".")
    if not name:
        return "upload"
    path = Path(name)
    suffix = path.suffix[:16]
    stem = path.stem[: _MAX_NAME_LENGTH - len(suffix)] or "upload"
    return f"{stem}{suffix}"


def validate_extension(file_name: str) -> str:
    """Return the lower-cased extension of ``file_name`` if it is supported.

    Raises:
        UnsupportedFormatError: The extension is not in
            :data:`SUPPORTED_EXTENSIONS`.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        shown = suffix or "(none)"
        msg = f"Unsupported format '{shown}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        raise UnsupportedFormatError(msg)
    return suffix


@asynccontextmanager
async def stage_upload(
    upload: UploadFile,
    *,
    upload_dir: Path,
    max_bytes: int,
) -> AsyncGenerator[StagedUpload]:
    """Validate ``upload``, write it to a private temp directory, and clean up.

    Args:
        upload: The multipart file.
        upload_dir: Parent directory for staging directories.
        max_bytes: Size limit; exceeding it aborts the copy.

    Yields:
        The staged file. It and its directory are deleted on exit.

    Raises:
        UnsupportedFormatError: Unsupported extension (checked before any
            byte is written).
        UploadTooLargeError: The file exceeds ``max_bytes``.
        EmptyUploadError: The file is empty.
    """
    file_name = safe_filename(upload.filename)
    validate_extension(file_name)

    upload_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="ingest-", dir=upload_dir))
    try:
        path = staging_dir / file_name
        size = 0
        with path.open("wb") as out:
            while chunk := await upload.read(_CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    msg = f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
                    raise UploadTooLargeError(msg)
                out.write(chunk)
        if size == 0:
            msg = f"Uploaded file '{file_name}' is empty."
            raise EmptyUploadError(msg)
        yield StagedUpload(path=path, file_name=file_name, size_bytes=size)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
