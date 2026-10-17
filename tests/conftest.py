"""Shared fixtures.

* ``settings_factory`` builds :class:`Settings` from explicit overrides only:
  every env var a field could read is cleared first and ``.env`` is ignored,
  so a developer's shell cannot change test outcomes.
* ``harness_factory`` boots the real app (lifespan included) through
  ``create_app(container_factory=...)`` with offline fakes in place of the
  embedding model, the LLM and Docling.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import pytest
from fastapi.testclient import TestClient

from docling_llamaindex_pipeline.api.deps import AppContainer
from docling_llamaindex_pipeline.app import create_app
from docling_llamaindex_pipeline.config import EmbeddingProvider, Settings
from docling_llamaindex_pipeline.providers import EmbeddingSelection
from docling_llamaindex_pipeline.rag_service import RagService
from tests.support.fakes import FakeParser, FixedLLM, KeywordEmbedding

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM


class SettingsFactory(Protocol):
    def __call__(self, **overrides: Any) -> Settings: ...


@dataclass
class Harness:
    client: TestClient
    container: AppContainer
    parser: FakeParser
    upload_dir: Path


class HarnessFactory(Protocol):
    def __call__(
        self,
        *,
        parser: FakeParser | None = None,
        embed_model: BaseEmbedding | None = None,
        llm: LLM | None = None,
        **settings_overrides: Any,
    ) -> Harness: ...


@pytest.fixture
def settings_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SettingsFactory:
    def _build(**overrides: Any) -> Settings:
        for name in Settings.model_fields:
            monkeypatch.delenv(name.upper(), raising=False)
        overrides.setdefault("temp_upload_dir", tmp_path / "uploads")
        return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]

    return _build


@pytest.fixture
def harness_factory(settings_factory: SettingsFactory) -> Iterator[HarnessFactory]:
    with ExitStack() as stack:

        def _build(
            *,
            parser: FakeParser | None = None,
            embed_model: BaseEmbedding | None = None,
            llm: LLM | None = None,
            **settings_overrides: Any,
        ) -> Harness:
            settings = settings_factory(**settings_overrides)
            fake_parser = parser or FakeParser()
            embedding = EmbeddingSelection(
                provider=EmbeddingProvider.OLLAMA,
                model_name="keyword-hash",
                model=embed_model or KeywordEmbedding(),
            )

            def _container(resolved: Settings) -> AppContainer:
                service = RagService(embedding=embedding, llm=llm or FixedLLM())
                return AppContainer(settings=resolved, rag_service=service, parser=fake_parser)

            app = create_app(settings, container_factory=_container)
            client: TestClient = stack.enter_context(TestClient(app, raise_server_exceptions=False))
            container: AppContainer = app.state.container
            return Harness(
                client=client,
                container=container,
                parser=fake_parser,
                upload_dir=settings.temp_upload_dir,
            )

        yield _build
