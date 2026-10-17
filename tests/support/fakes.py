"""Deterministic, network-free stand-ins for embeddings, LLMs and the parser."""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from typing import TYPE_CHECKING, Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import (
    llm_completion_callback,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.schema import BaseNode, TextNode
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from llama_index.core.llms import CompletionResponseGen

_DIM = 64
_TOKEN = re.compile(r"\w+")


def _connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("connection refused", request=httpx.Request("POST", "http://model"))


class KeywordEmbedding(BaseEmbedding):
    """Hashed bag-of-words embedding: texts sharing words score higher, deterministically."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="keyword-hash", **kwargs)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * _DIM
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest) % _DIM] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)


class UnreachableEmbedding(BaseEmbedding):
    """Embedding whose server refuses every connection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="unreachable", **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        raise _connect_error()

    def _get_text_embedding(self, text: str) -> list[float]:
        raise _connect_error()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        raise _connect_error()


class BrokenEmbedding(UnreachableEmbedding):
    """Embedding that fails for a non-network reason."""

    def _get_text_embedding(self, text: str) -> list[float]:
        msg = "dimension mismatch"
        raise ValueError(msg)


class FixedLLM(CustomLLM):
    """LLM that always answers ``answer`` and records the prompts it saw.

    ``failure`` makes every call raise it instead; ``delay`` makes async calls
    sleep first (for timeout tests).
    """

    answer: str = "fixed answer"
    delay: float = 0.0
    prompts: list[str] = Field(default_factory=list[str])
    failure: Exception | None = None

    @property
    def metadata(self) -> LLMMetadata:
        """Plain completion model with a generous window."""
        return LLMMetadata(context_window=32_000, num_output=256, model_name="fixed")

    @llm_completion_callback()  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Return the fixed answer (or raise the configured failure)."""
        if self.failure is not None:
            raise self.failure
        self.prompts.append(prompt)
        return CompletionResponse(text=self.answer)

    @llm_completion_callback()  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async variant honouring ``delay``."""
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.complete(prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming is never used by the service."""
        raise NotImplementedError


def unreachable_llm() -> FixedLLM:
    """LLM whose server refuses every connection."""
    return FixedLLM(failure=_connect_error())


def text_node(text: str, **metadata: Any) -> TextNode:
    """Build a node shaped like ``DoclingNodeParser`` output."""
    meta: dict[str, Any] = {"doc_items": [], "headings": []}
    meta.update(metadata)
    return TextNode(text=text, metadata=meta)


class FakeParser:
    """:class:`DocumentParser` returning canned nodes and recording what it saw.

    Args:
        nodes: Nodes to return (copied, with fresh IDs, on every call).
        factory: Produces the nodes per call instead of ``nodes``.
        error: Raised instead of returning, when set.
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode] = (),
        *,
        factory: Callable[[], list[BaseNode]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._nodes = nodes
        self._factory = factory
        self.error = error
        self.calls: list[Path] = []
        self.file_existed: list[bool] = []
        self.contents: list[bytes] = []

    def parse(self, path: Path) -> list[BaseNode]:
        """Record ``path`` and return the configured nodes."""
        self.calls.append(path)
        self.file_existed.append(path.exists())
        self.contents.append(path.read_bytes())
        if self.error is not None:
            raise self.error
        if self._factory is not None:
            return self._factory()
        return [
            node.model_copy(update={"id_": str(uuid.uuid4())}, deep=True) for node in self._nodes
        ]
