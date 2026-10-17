"""Multi-format document ingestion & grounded RAG API (Docling + LlamaIndex).

The application lives in :mod:`docling_llamaindex_pipeline.app`; this module
only exposes the version and the ``docling-llamaindex-pipeline`` console
entry point.
"""

from __future__ import annotations

__all__ = ["__version__", "main"]

__version__ = "0.1.0"


def main() -> None:
    """Serve the API with uvicorn, bound to ``API_HOST`` / ``API_PORT``."""
    import uvicorn  # Deferred: keeps `import docling_llamaindex_pipeline` light.

    from docling_llamaindex_pipeline.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "docling_llamaindex_pipeline.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )
