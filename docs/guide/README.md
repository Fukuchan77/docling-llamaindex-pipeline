# 利用ガイド: セットアップから基本的な使い方まで

Docling × LlamaIndex × Ollama による、完全ローカルで動くグラウンディング付き RAG API
（本リポジトリ）の導入ガイドです。各手順には、詳細を確認できる公式ガイドへのリンクを付けています。

| # | ドキュメント | 内容 |
| :-- | :--- | :--- |
| 1 | [セットアップ](01-setup.md) | mise / uv / Python 3.14、Ollama と IBM Granite モデル、Tesseract、Docling モデル、`.env` |
| 2 | [各コンポーネントの基本](02-components.md) | Docling 単体（CLI / SDK / Granite-Docling）、LlamaIndex × Ollama、Docling × LlamaIndex のチャンク分割 |
| 3 | [API の使い方](03-api-usage.md) | サーバ起動、取り込み、質問、グラウンディング座標、埋め込み切替、エラーとトラブルシュート |

仕様・基本設計は [`../SPEC-RAG-2026-001.md`](../SPEC-RAG-2026-001.md) を参照してください。

## 採用モデル（IBM Granite に統一）

| 役割 | モデル | 実行場所 | 公式情報 |
| :--- | :--- | :--- | :--- |
| 回答生成 LLM | `granite4.2:latest` | Ollama | [Ollama ライブラリ](https://ollama.com/library/granite4.2) / [IBM Granite 4.2](https://www.ibm.com/granite/docs/models/granite4-2) |
| 埋め込み | `granite-embedding:278m`（多言語・768 次元・512 トークン） | Ollama | [Ollama ライブラリ](https://ollama.com/library/granite-embedding) / [Hugging Face](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) |
| チャンク分割用トークナイザ | `ibm-granite/granite-embedding-278m-multilingual` | ローカル（初回のみ取得） | [Hugging Face](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) |
| 文書解析（標準） | Docling レイアウトモデル + TableFormer | ローカル | [Docling](https://docling-project.github.io/docling/) |
| 文書解析（VLM・任意） | Granite-Docling-258M | ローカル | [IBM 発表](https://www.ibm.com/jp-ja/new/announcements/granite-docling-end-to-end-document-conversion) / [Hugging Face](https://huggingface.co/ibm-granite/granite-docling-258M) |
| OCR | Tesseract（既定）または EasyOCR | ローカル | [Docling OCR](https://docling-project.github.io/docling/concepts/OCR/) |

**中国製の LLM・埋め込み・OCR モデルは使用しません。** この方針は
`tests/unit/test_model_policy.py` で検査しています。

- 既定値（`config.py`）、`src/`・`examples/` のコード、`.env.example`、`mise.toml`、CI 設定に該当するモデル名が含まれていると、テストが失敗します。
- Docling は OCR を自動選択させると、Linux では RapidOCR（PaddleOCR 由来のモデル）を選ぶことがあります。
  そのため OCR エンジンを `DOCLING_OCR_ENGINE` で明示しています。
- Docling の VLM プリセットには他社モデルも含まれます。そのため Granite-Docling を明示的に指定します。

## 公式ガイド・リファレンス

### LlamaIndex
- [LlamaIndex Developer Documentation](https://developers.llamaindex.ai/): Python フレームワークの総合ポータル
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index): 本体と統合パッケージ（`llama-index-llms-ollama`、`llama-index-readers-docling` など）
- [LlamaIndex とは（IBM Think）](https://www.ibm.com/jp-ja/think/topics/llamaindex): Reader・インデックス・リトリーバーなど主要概念の日本語解説
- [インストール](https://developers.llamaindex.ai/python/framework/getting_started/installation/) / [ローカル LLM でのスターター](https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local/)（例で使われているモデルは本プロジェクトの採用モデルに読み替えてください）
- [Ollama LLM 連携](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/) / [Ollama Embedding 連携](https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding/)
- [Docling Reader / Node Parser デモ](https://developers.llamaindex.ai/python/examples/data_connectors/doclingreaderdemo/) / [Node Parser モジュール](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)

### Docling / Granite-Docling
- [Docling 公式ドキュメント](https://docling-project.github.io/docling/) / [Docling GitHub](https://github.com/docling-project/docling)
- [IBM Granite-Docling 発表・技術解説](https://www.ibm.com/jp-ja/new/announcements/granite-docling-end-to-end-document-conversion) / [IBM Granite Docs: Docling](https://www.ibm.com/granite/docs/models/docling)
- [インストール](https://docling-project.github.io/docling/getting_started/installation/) / [使い方](https://docling-project.github.io/docling/usage/) / [対応フォーマット](https://docling-project.github.io/docling/usage/supported_formats/) / [CLI リファレンス](https://docling-project.github.io/docling/reference/cli/)
- [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/) / [チャンク分割](https://docling-project.github.io/docling/concepts/chunking/) / [HybridChunker の例](https://docling-project.github.io/docling/examples/hybrid_chunking/) / [OCR](https://docling-project.github.io/docling/concepts/OCR/) / [連携一覧](https://docling-project.github.io/docling/integrations/)

### Ollama / IBM Granite
- [Ollama ダウンロード](https://ollama.com/download) / [Ollama ドキュメント](https://docs.ollama.com/) / [Ollama GitHub](https://github.com/ollama/ollama)
- [Granite を Ollama で動かす（IBM）](https://www.ibm.com/granite/docs/run/granite-with-ollama-linux) / [Granite Embedding 製品ガイド（IBM）](https://www.ibm.com/architectures/product-guides/granite-embedding) / [ibm-granite（Hugging Face）](https://huggingface.co/ibm-granite)

### 開発ツール
- [mise: Getting Started](https://mise.jdx.dev/getting-started.html) / [mise: Tasks](https://mise.jdx.dev/tasks/)
- [uv: Installation](https://docs.astral.sh/uv/getting-started/installation/) / [uv: Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) / [uv: Optional dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies)
- [FastAPI: Request Files](https://fastapi.tiangolo.com/tutorial/request-files/) / [Tesseract: Installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
