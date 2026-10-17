"""The in-memory index and the operations on it (spec §3, §4.4).

:class:`RagService` owns the ``VectorStoreIndex``, the active embedding
adapter and the synthesis LLM. All state changes go through one
:class:`asyncio.Lock`:

* ingestion embeds and inserts under the lock, so a concurrent hot-swap can
  never leave vectors from the old model in the new index (or count a
  document into an index that has just been discarded);
* queries take no lock: they capture the current index and LLM once and run
  against that snapshot, so a hot-swap that lands mid-query does not affect
  them.

Network failures of the embedding / LLM server become
:class:`ModelServerUnavailableError` (HTTP 503); anything else becomes a 500.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.vector_stores.simple import SimpleVectorStore

from docling_llamaindex_pipeline.errors import (
    IndexingError,
    ModelServerUnavailableError,
    NoDocumentsIndexedError,
    QueryExecutionError,
    is_model_server_unreachable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.core.llms import LLM
    from llama_index.core.schema import BaseNode, NodeWithScore

    from docling_llamaindex_pipeline.config import EmbeddingProvider
    from docling_llamaindex_pipeline.providers import EmbeddingSelection

__all__ = ["IndexStats", "QueryResult", "RagService"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Snapshot of the index for ``/health``.

    Attributes:
        documents: Documents ingested since the index was (re)created.
        nodes: Nodes in the index.
        embedding_provider: Active embedding provider.
        embedding_model: Active embedding model ID.
    """

    documents: int
    nodes: int
    embedding_provider: EmbeddingProvider
    embedding_model: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Outcome of a query.

    Attributes:
        answer: The synthesized answer.
        sources: Retrieved nodes with similarity scores, best first.
        embedding_provider: Provider of the index that answered.
    """

    answer: str
    sources: Sequence[NodeWithScore]
    embedding_provider: EmbeddingProvider


def _new_index(embedding: EmbeddingSelection) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
    return VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=embedding.model,
    )


class RagService:
    """Owns the vector index and runs ingestion, queries and hot-swaps.

    Args:
        embedding: The embedding adapter active at startup.
        llm: The LLM used to synthesize answers.
    """

    def __init__(self, *, embedding: EmbeddingSelection, llm: LLM) -> None:
        self._lock = asyncio.Lock()
        self._llm = llm
        self._embedding = embedding
        self._index = _new_index(embedding)
        self._documents = 0
        self._nodes = 0

    @property
    def stats(self) -> IndexStats:
        """Current document / node counts and the active embedding model."""
        return IndexStats(
            documents=self._documents,
            nodes=self._nodes,
            embedding_provider=self._embedding.provider,
            embedding_model=self._embedding.model_name,
        )

    async def add_document(self, nodes: Sequence[BaseNode]) -> EmbeddingProvider:
        """Embed ``nodes`` with the active model and insert them as one document.

        Args:
            nodes: The parsed nodes of one document. Must be non-empty.

        Returns:
            The embedding provider the nodes were embedded with.

        Raises:
            ModelServerUnavailableError: The embedding server is unreachable.
            IndexingError: Embedding or insertion failed otherwise.
        """
        async with self._lock:
            try:
                await self._index.ainsert_nodes(list(nodes))
            except Exception as exc:  # Re-raised as a typed domain error.
                if is_model_server_unreachable(exc):
                    msg = f"Model server unreachable (embedding provider '{self._embedding.provider}')."
                    raise ModelServerUnavailableError(msg) from exc
                msg = "Indexing failed while embedding the document."
                raise IndexingError(msg) from exc
            self._documents += 1
            self._nodes += len(nodes)
            return self._embedding.provider

    async def query(self, text: str, *, top_k: int) -> QueryResult:
        """Retrieve the ``top_k`` most similar nodes and synthesize an answer.

        Args:
            text: The user's question.
            top_k: Number of nodes to retrieve.

        Returns:
            The answer and its source nodes.

        Raises:
            NoDocumentsIndexedError: Nothing has been ingested yet.
            ModelServerUnavailableError: The embedding or LLM server is
                unreachable.
            QueryExecutionError: Retrieval or synthesis failed otherwise.
        """
        index, embedding, nodes = self._index, self._embedding, self._nodes
        if nodes == 0:
            msg = "No documents have been indexed yet. Please call /api/v1/ingest first."
            raise NoDocumentsIndexedError(msg)

        # `llm` is typed `LLMType`, whose LangChain arm is unresolvable without langchain.
        engine = index.as_query_engine(llm=self._llm, similarity_top_k=top_k)  # pyright: ignore[reportUnknownMemberType]
        try:
            response = await engine.aquery(text)
        except Exception as exc:  # Re-raised as a typed domain error.
            if is_model_server_unreachable(exc):
                msg = "Model server unreachable."
                raise ModelServerUnavailableError(msg) from exc
            msg = "Query execution failed."
            raise QueryExecutionError(msg) from exc

        if not isinstance(response, Response):  # pragma: no cover — streaming is never enabled.
            msg = f"Unexpected response type {type(response).__name__}."
            raise QueryExecutionError(msg)
        return QueryResult(
            answer=response.response or "",
            sources=response.source_nodes,
            embedding_provider=embedding.provider,
        )

    async def switch_embedding(self, embedding: EmbeddingSelection) -> None:
        """Activate ``embedding`` and start over with an empty index.

        Vectors from different models are not comparable (and usually differ
        in dimension), so the existing index is discarded; documents must be
        re-ingested.

        Args:
            embedding: The new embedding adapter.
        """
        async with self._lock:
            self._embedding = embedding
            self._index = _new_index(embedding)
            self._documents = 0
            self._nodes = 0
        logger.info(
            "Embedding provider switched to %s (%s); index reset.",
            embedding.provider,
            embedding.model_name,
        )
