"""Domain exceptions and their HTTP mapping (spec §7.1).

Every error this service raises on purpose is a :class:`PipelineError`
subclass carrying its HTTP status. :func:`install_exception_handlers`
registers one handler that renders them as FastAPI's usual
``{"detail": "..."}`` body, so route handlers raise domain errors and never
build HTTP responses themselves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import httpx
from fastapi import status
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

__all__ = [
    "DocumentParsingError",
    "EmptyUploadError",
    "IndexingError",
    "ModelServerUnavailableError",
    "NoDocumentsIndexedError",
    "NoExtractableContentError",
    "PipelineError",
    "ProviderConfigurationError",
    "QueryExecutionError",
    "QueryTimeoutError",
    "UnauthorizedError",
    "UnsupportedFormatError",
    "UploadTooLargeError",
    "install_exception_handlers",
    "is_model_server_unreachable",
]

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base class for errors that map to a specific HTTP status."""

    status_code: ClassVar[int] = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFormatError(PipelineError):
    """The uploaded file's extension is not in the supported set."""

    status_code = status.HTTP_400_BAD_REQUEST


class EmptyUploadError(PipelineError):
    """The uploaded file has no content."""

    status_code = status.HTTP_400_BAD_REQUEST


class NoExtractableContentError(PipelineError):
    """Docling parsed the file but produced no text chunks to index."""

    status_code = status.HTTP_400_BAD_REQUEST


class NoDocumentsIndexedError(PipelineError):
    """A query was issued before any document was indexed."""

    status_code = status.HTTP_400_BAD_REQUEST


class ProviderConfigurationError(PipelineError):
    """An unknown provider, a missing credential, or an uninstalled extra."""

    status_code = status.HTTP_400_BAD_REQUEST


class UnauthorizedError(PipelineError):
    """The admin token is missing or wrong."""

    status_code = status.HTTP_401_UNAUTHORIZED


class UploadTooLargeError(PipelineError):
    """The upload exceeds ``MAX_UPLOAD_SIZE_MB``."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class DocumentParsingError(PipelineError):
    """Docling failed to convert the document (corrupt, encrypted, ...)."""


class IndexingError(PipelineError):
    """Embedding or inserting the parsed nodes failed for a non-network reason."""


class QueryExecutionError(PipelineError):
    """Retrieval or synthesis failed for a non-network reason."""


class ModelServerUnavailableError(PipelineError):
    """The embedding / LLM server could not be reached."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class QueryTimeoutError(PipelineError):
    """The query exceeded ``QUERY_TIMEOUT_SECONDS``."""

    status_code = status.HTTP_504_GATEWAY_TIMEOUT


_UNREACHABLE_TYPES: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)
"""Exception types that mean "the model server is unreachable".

``httpx.TransportError`` covers connect / read timeouts and refused
connections from every client built on httpx (Ollama, OpenAI, watsonx). The
``ollama`` client re-raises a refused connection as the builtin
``ConnectionError``, and SDKs that wrap the transport error (e.g.
``openai.APIConnectionError``) keep it as ``__cause__``, which
:func:`is_model_server_unreachable` follows.
"""


def is_model_server_unreachable(exc: BaseException) -> bool:
    """Return whether ``exc`` (or anything in its cause chain) is a network failure.

    Args:
        exc: The exception raised by an embedding or LLM call.

    Returns:
        ``True`` when the failure is a transport-level error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, _UNREACHABLE_TYPES):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


async def _pipeline_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a :class:`PipelineError` as ``{"detail": ...}`` at its status code."""
    del request
    assert isinstance(exc, PipelineError)  # noqa: S101 — registration guarantees the type.
    if isinstance(exc, ModelServerUnavailableError):
        # Expected while a model server restarts; the cause is already in the detail.
        logger.warning("%s: %s (%r)", type(exc).__name__, exc.detail, exc.__cause__)
    elif exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error("%s: %s", type(exc).__name__, exc.detail, exc_info=exc.__cause__)
    headers = {"WWW-Authenticate": "X-Admin-Token"} if isinstance(exc, UnauthorizedError) else None
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=headers)


def install_exception_handlers(app: FastAPI) -> None:
    """Register the :class:`PipelineError` handler on ``app``."""
    app.add_exception_handler(PipelineError, _pipeline_error_handler)
