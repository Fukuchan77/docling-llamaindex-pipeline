"""``POST /api/v1/config/embedding`` — embedding hot-swap (spec §4.4).

The new adapter is built *before* anything changes, so an unknown provider,
a missing credential or an uninstalled extra answers HTTP 400 and leaves the
current index untouched. On success the index is reset.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from docling_llamaindex_pipeline.api.deps import ContainerDep, require_admin
from docling_llamaindex_pipeline.config import EmbeddingProvider
from docling_llamaindex_pipeline.errors import ProviderConfigurationError
from docling_llamaindex_pipeline.providers import build_embedding
from docling_llamaindex_pipeline.schemas import EmbeddingSwitchRequest, EmbeddingSwitchResponse

router = APIRouter(prefix="/api/v1/config", tags=["config"])


def _parse_provider(value: str) -> EmbeddingProvider:
    try:
        return EmbeddingProvider(value.strip().lower())
    except ValueError:
        allowed = [p.value for p in EmbeddingProvider]
        msg = f"Unknown embedding provider '{value}'. Allowed: {allowed}"
        raise ProviderConfigurationError(msg) from None


@router.post(
    "/embedding",
    summary="Hot-swap the active embedding provider",
    dependencies=[Depends(require_admin)],
    responses={
        400: {"description": "Unknown provider, missing credential, or missing extra."},
        401: {"description": "ADMIN_API_KEY is set and X-Admin-Token is missing or wrong."},
    },
)
async def switch_embedding_provider(
    body: EmbeddingSwitchRequest,
    container: ContainerDep,
) -> EmbeddingSwitchResponse:
    """Activate another embedding provider and reset the index."""
    provider = _parse_provider(body.provider)
    try:
        embedding = build_embedding(provider, container.settings, model_name=body.model_name)
    except ProviderConfigurationError:
        raise
    except Exception as exc:  # SDK constructors (e.g. watsonx auth) raise anything.
        msg = f"Failed to switch embedding provider: {type(exc).__name__}."
        raise ProviderConfigurationError(msg) from exc

    await container.rag_service.switch_embedding(embedding)
    return EmbeddingSwitchResponse(
        active_provider=embedding.provider,
        embedding_model=embedding.model_name,
        message="Embedding model changed successfully. Please re-index documents.",
    )
