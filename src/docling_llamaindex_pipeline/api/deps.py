"""FastAPI ``Depends`` providers.

The application's long-lived objects are created by the lifespan and kept in
an :class:`AppContainer` on ``app.state``; these providers hand them to route
handlers. Tests inject their own container through ``create_app`` instead of
overriding dependencies one by one.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Header, Request

from docling_llamaindex_pipeline.errors import UnauthorizedError

if TYPE_CHECKING:
    from docling_llamaindex_pipeline.config import Settings
    from docling_llamaindex_pipeline.parsing import DocumentParser
    from docling_llamaindex_pipeline.rag_service import RagService

__all__ = [
    "AppContainer",
    "ContainerDep",
    "get_container",
    "require_admin",
]


@dataclass(slots=True)
class AppContainer:
    """Long-lived objects shared by all requests.

    Attributes:
        settings: The frozen runtime configuration.
        rag_service: The vector index and query engine.
        parser: Converts uploads into nodes.
        parse_slots: Bounds concurrent Docling conversions to
            ``settings.max_concurrent_ingestions``.
    """

    settings: Settings
    rag_service: RagService
    parser: DocumentParser
    parse_slots: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.parse_slots = asyncio.Semaphore(self.settings.max_concurrent_ingestions)


def get_container(request: Request) -> AppContainer:
    """Return the :class:`AppContainer` the lifespan stored on ``app.state``."""
    container: AppContainer = request.app.state.container
    return container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


def require_admin(
    container: ContainerDep,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    """Guard admin-only routes with ``ADMIN_API_KEY`` when one is configured.

    Args:
        container: The application container (for the configured key).
        x_admin_token: The ``X-Admin-Token`` request header.

    Raises:
        UnauthorizedError: A key is configured and the header is missing or
            does not match.
    """
    expected = container.settings.admin_api_key
    if expected is None:
        return
    if x_admin_token is None or not secrets.compare_digest(
        x_admin_token.encode(), expected.get_secret_value().encode()
    ):
        msg = "A valid X-Admin-Token header is required."
        raise UnauthorizedError(msg)
