"""FastAPI application factory (spec §2.1).

:func:`create_app` composes the application and contains no request logic:

1. settings are read once (fail-fast validation, see :mod:`.config`);
2. middleware: a ``Content-Length`` guard that rejects oversized bodies before
   Starlette spools them to disk, and CORS restricted to
   ``CORS_ALLOW_ORIGINS`` (disabled when empty);
3. the lifespan builds the embedding adapter, the LLM, the Docling parser and
   the :class:`RagService`, and stores them in an :class:`AppContainer`;
4. routers and the domain-error handler are registered.

Run it with uvicorn's factory mode (no import-time side effects)::

    uvicorn docling_llamaindex_pipeline.app:create_app --factory
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from docling_llamaindex_pipeline import __version__
from docling_llamaindex_pipeline.api.deps import AppContainer
from docling_llamaindex_pipeline.api.routes import embedding, health, ingest, query
from docling_llamaindex_pipeline.config import Settings, get_settings
from docling_llamaindex_pipeline.errors import install_exception_handlers
from docling_llamaindex_pipeline.parsing import DoclingDocumentParser, build_document_converter
from docling_llamaindex_pipeline.providers import build_embedding, build_llm
from docling_llamaindex_pipeline.rag_service import RagService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = ["create_app"]

logger = logging.getLogger("docling_llamaindex_pipeline")

_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
"""Allowance for multipart boundaries and headers on top of the file itself."""


class ContentLengthLimitMiddleware:
    """Reject requests whose declared ``Content-Length`` exceeds a limit (HTTP 413).

    Starlette spools a multipart body to disk before the route runs, so the
    per-file check in :func:`.uploads.stage_upload` alone would still accept
    the whole body. Bodies sent without ``Content-Length`` (chunked) fall
    through to that per-file check.

    Args:
        app: The wrapped ASGI app.
        max_body_bytes: Largest accepted ``Content-Length``.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Short-circuit oversized HTTP requests; pass everything else through."""
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name == b"content-length":
                    if not value.isdigit() or int(value) > self.max_body_bytes:
                        response = JSONResponse(
                            {"detail": "Request body exceeds the upload limit."},
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        )
                        await response(scope, receive, send)
                        return
                    break
        await self.app(scope, receive, send)


def _default_container(settings: Settings) -> AppContainer:
    embedding_selection = build_embedding(settings.embedding_provider, settings)
    rag_service = RagService(embedding=embedding_selection, llm=build_llm(settings))
    parser = DoclingDocumentParser(build_document_converter(settings))
    return AppContainer(settings=settings, rag_service=rag_service, parser=parser)


def _configure_logging(level: str) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: Callable[[Settings], AppContainer] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration to use; defaults to :func:`get_settings`.
        container_factory: Builds the :class:`AppContainer` at startup.
            Defaults to the production wiring; tests pass one returning
            fakes so no model server or Docling model is needed.

    Returns:
        The configured application.
    """
    resolved = settings or get_settings()
    factory = container_factory or _default_container

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        _configure_logging(resolved.log_level)
        app.state.container = factory(resolved)
        stats = app.state.container.rag_service.stats
        logger.info(
            "Pipeline ready: embeddings=%s (%s), llm=%s.",
            stats.embedding_provider,
            stats.embedding_model,
            resolved.llm_provider,
        )
        yield

    app = FastAPI(
        title="Docling x LlamaIndex Ingestion & RAG API",
        description=(
            "Multi-format document parser (PDF, Office, HTML, CSV, XML) with "
            "bounding-box grounded retrieval-augmented answers."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        ContentLengthLimitMiddleware,
        max_body_bytes=resolved.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    if resolved.cors_allow_origins:
        wildcard = "*" in resolved.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_allow_origins,
            # Credentials with a wildcard origin are forbidden by the CORS spec.
            allow_credentials=not wildcard,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Admin-Token"],
        )
    install_exception_handlers(app)
    for module in (health, ingest, query, embedding):
        app.include_router(module.router)
    return app
