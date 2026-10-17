"""Smoke-test the Ollama models through LlamaIndex (no Docling, no API server).

Usage::

    uv run python examples/02_ollama_basic.py

Model names come from the same settings the API uses (``OLLAMA_LLM_MODEL`` /
``OLLAMA_EMBED_MODEL`` / ``OLLAMA_BASE_URL``; defaults ``granite4.2:latest`` and
``granite-embedding:278m``), so this checks exactly what the server will call.

Docs:
* https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/
* https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding/
"""

from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from docling_llamaindex_pipeline.config import get_settings


def main() -> None:
    """Embed one sentence and ask one question."""
    settings = get_settings()

    embed_model = OllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    vector = embed_model.get_text_embedding("DoclingとLlamaIndexによるローカルRAGパイプライン")
    print(f"embedding: {settings.ollama_embed_model} -> {len(vector)} dims")

    llm = Ollama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_request_timeout,
        temperature=0.1,
    )
    response = llm.complete(
        "RAGで文書構造(見出し階層や表構造)を保持するメリットを日本語で3行で説明してください。"
    )
    print(f"\n[{settings.ollama_llm_model}]\n{response.text.strip()}")


if __name__ == "__main__":
    main()
