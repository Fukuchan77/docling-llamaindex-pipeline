# 1. セットアップ

ゴールは、`mise run check` が通り、API サーバの `/health` が応答する状態です。

| 項目 | 内容 |
| :--- | :--- |
| 対象 OS | Linux / macOS（Windows は WSL2 推奨） |
| 必要リソース | CPU 8 コア / RAM 16 GB 以上（仕様 §6.2） |
| ディスク | 依存パッケージで約 6 GB（CUDA 版 torch を含む）、モデルで数 GB |

[← ガイド目次](README.md) ・ [次へ: 各コンポーネントの基本 →](02-components.md)

---

## 1.1 mise と uv（Python 3.14）

> 公式: [mise Getting Started](https://mise.jdx.dev/getting-started.html) ・ [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)

`mise` がリポジトリの [`mise.toml`](../../mise.toml) を読み、Python 3.14 と uv 0.13 を固定バージョンで用意します。

```bash
curl https://mise.run | sh        # mise 本体（他の方法は公式ガイド参照）
git clone https://github.com/Fukuchan77/docling-llamaindex-pipeline.git
cd docling-llamaindex-pipeline
mise trust && mise install

python --version   # Python 3.14.x
uv --version       # uv 0.13.x
```

## 1.2 依存パッケージ

> 公式: [uv: Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) ・ [uv: Optional dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies) ・ [Docling Installation](https://docling-project.github.io/docling/getting_started/installation/) ・ [LlamaIndex Installation](https://developers.llamaindex.ai/python/framework/getting_started/installation/)

```bash
mise run setup     # = uv sync --all-extras --all-groups && uv run pre-commit install
```

| 用途 | コマンド |
| :--- | :--- |
| 開発（テスト一式を含む） | `mise run setup` |
| Ollama だけで運用 | `uv sync --no-dev` |
| OpenAI へ切り替える可能性がある | `uv sync --no-dev --extra openai` |
| 回答生成に Anthropic を使う可能性がある | `uv sync --no-dev --extra anthropic` |
| IBM watsonx.ai へ切り替える可能性がある | `uv sync --no-dev --extra ibm` |
| Apple Silicon で Granite-Docling（VLM）を MLX で高速に動かす | `uv sync --no-dev --extra mlx`（[2.2](02-components.md#22-docling-granite-docling-258mvlm-で変換する)） |

> **Note:** `mlx` extra は `mlx-vlm` を Apple Silicon（macOS / arm64）にだけインストールします。他の環境では何も入らないため、`mise run setup`（`--all-extras`）でそのまま使えます。

依存関係は `uv.lock` で固定されています。追加・更新は `uv add <pkg>` で行い、`pip install` は使いません。

主な依存パッケージは次のとおりです（[`pyproject.toml`](../../pyproject.toml)）。

| パッケージ | 役割 | 公式 |
| :--- | :--- | :--- |
| `docling` | 文書解析（レイアウト・表・座標） | [Docs](https://docling-project.github.io/docling/) |
| `llama-index-core` | インデックス・検索・回答合成 | [Docs](https://developers.llamaindex.ai/) |
| `llama-index-readers-docling` / `llama-index-node-parser-docling` | Docling ⇔ LlamaIndex 変換 | [デモ](https://developers.llamaindex.ai/python/examples/data_connectors/doclingreaderdemo/) |
| `llama-index-llms-ollama` / `llama-index-embeddings-ollama` | Ollama 連携 | [LLM](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/) / [Embedding](https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding/) |
| `fastapi` / `uvicorn` | REST API | [FastAPI](https://fastapi.tiangolo.com/) |

> **Note:** Docling は PyPI から CUDA 版 torch を取り込みます。GPU がない環境でも動作しますが、初回の `uv sync` には数分かかります。

## 1.3 Ollama と IBM Granite モデル

> 公式: [Ollama ダウンロード](https://ollama.com/download) ・ [Ollama ドキュメント](https://docs.ollama.com/) ・ [Granite を Ollama で動かす（IBM）](https://www.ibm.com/granite/docs/run/granite-with-ollama-linux)

```bash
# Ollama をインストールして起動（macOS / Windows はアプリを起動するだけ）
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # サービスとして起動済みなら不要

# 回答生成用 LLM: IBM Granite 4.2
ollama pull granite4.2:latest
# 埋め込み: IBM Granite Embedding 278M（多言語・日本語対応・768 次元・512 トークン）
ollama pull granite-embedding:278m

ollama list
```

| モデル | 公式 |
| :--- | :--- |
| `granite4.2:latest` | [Ollama ライブラリ](https://ollama.com/library/granite4.2) / [IBM Granite 4.2](https://www.ibm.com/granite/docs/models/granite4-2) |
| `granite-embedding:278m` | [Ollama ライブラリ](https://ollama.com/library/granite-embedding) / [Hugging Face](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) / [IBM 製品ガイド](https://www.ibm.com/architectures/product-guides/granite-embedding) |

> **Tip:** `latest` タグの中身は Granite 4.2 の更新に合わせて変わります。本番環境で結果の再現性が必要な場合は、[ライブラリページ](https://ollama.com/library/granite4.2/tags)に載っているサイズ別のタグを `OLLAMA_LLM_MODEL` に指定してください。

> **Note:** 埋め込みモデルを変更した場合は、既存インデックスとの互換性がなくなります。埋め込みを切り替えるとインデックスは初期化され、文書の再取り込みが必要です（[3. API の使い方](03-api-usage.md#35-埋め込みプロバイダの切り替え--post-apiv1configembedding)）。

## 1.4 OCR（Tesseract）

> 公式: [Docling OCR](https://docling-project.github.io/docling/concepts/OCR/) ・ [Tesseract Installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)

スキャン PDF や画像内の文字を読み取るため、OCR エンジンに Tesseract を使います（`DOCLING_OCR_ENGINE=tesseract`）。
Docling の自動選択に任せると PaddleOCR 由来のモデルが選ばれることがあるため、エンジンを明示しています。

```bash
# Debian / Ubuntu
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
# macOS (Homebrew)
brew install tesseract tesseract-lang

tesseract --list-langs   # jpn と eng が表示されること
```

| 状況 | 設定 |
| :--- | :--- |
| Tesseract を入れられない | `DOCLING_OCR_ENGINE=easyocr`（`uv add easyocr` が必要） |
| デジタル生成の PDF だけを扱う | `DOCLING_DO_OCR=false` で OCR を無効化 |

## 1.5 Docling のモデルとチャンク用トークナイザ

> 公式: [Docling Installation](https://docling-project.github.io/docling/getting_started/installation/) ・ [CLI リファレンス](https://docling-project.github.io/docling/reference/cli/) ・ [Transformers オフラインモード](https://huggingface.co/docs/transformers/installation#offline-mode)

Hugging Face Hub から自動で取得されるものが 2 つあります。

| 対象 | 取得されるタイミング |
| :--- | :--- |
| Docling のレイアウト / TableFormer モデル | 最初の PDF を解析したとき |
| `granite-embedding:278m` 用トークナイザ（`ibm-granite/granite-embedding-278m-multilingual`） | サーバ起動時 |

ネットワークに接続できる環境では、特に作業は不要です。

**エアギャップ環境**では、接続できるマシンで事前に取得し、取得物を持ち込みます。

```bash
# Docling モデル（標準パイプラインに必要なものだけを指定）
uv run docling-tools models download layout tableformer -o ./models/docling
# Granite-Docling（VLM）も使う場合
uv run docling-tools models download granitedocling -o ./models/docling

# チャンク分割用トークナイザ（~/.cache/huggingface に保存される）
uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ibm-granite/granite-embedding-278m-multilingual')"
```

持ち込み先で次のように設定します。

```ini
DOCLING_ARTIFACTS_PATH=./models/docling
HF_HUB_OFFLINE=1
```

> **Warning:** `docling-tools models download` を引数なしで実行すると、既定セットとして RapidOCR などのモデルも取得されます。必要なモデル名を明示してください。

## 1.6 `.env`

> 公式: [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

```bash
cp .env.example .env
```

既定値のままで、ローカルの Ollama と Granite モデルを使って動作します。主な項目は次のとおりです（全項目は [`.env.example`](../../.env.example)）。

| 変数 | 既定値 | 説明 |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama の URL（`/v1` は付けない） |
| `OLLAMA_LLM_MODEL` | `granite4.2:latest` | 回答生成モデル |
| `OLLAMA_EMBED_MODEL` | `granite-embedding:278m` | 埋め込みモデル |
| `CHUNKER` / `CHUNK_MAX_TOKENS` | `hybrid` / `512` | 埋め込みの上限に合わせたチャンク分割 |
| `DOCLING_OCR_ENGINE` / `DOCLING_OCR_LANG` | `tesseract` / `["iso:jpn","iso:eng"]` | OCR エンジンと言語 |
| `MAX_UPLOAD_SIZE_MB` | `50` | アップロード上限 |
| `CORS_ALLOW_ORIGINS` | `[]` | 許可するフロントエンドのオリジン（空なら CORS 無効） |
| `ADMIN_API_KEY` | （未設定） | 埋め込み切替 API の保護。`APP_ENV=production` では必須 |

設定は起動時に検証されます。たとえば `EMBEDDING_PROVIDER=openai` なのに `OPENAI_API_KEY` が未設定の場合、サーバは起動に失敗します。

## 1.7 動作確認

> 公式: [mise Tasks](https://mise.jdx.dev/tasks/)

```bash
mise run check                         # lint / format / 型チェック / テスト（Ollama 不要）
uv run python examples/02_ollama_basic.py   # Ollama 上の Granite モデルに実際に問い合わせる
mise run dev                           # http://127.0.0.1:8000/docs
curl -s localhost:8000/health
```

`/health` が次のように返れば準備完了です。

```json
{"status":"healthy","indexed_documents":0,"indexed_nodes":0,
 "active_embedding_provider":"ollama","embedding_model":"granite-embedding:278m"}
```

Ollama・Tesseract・Docling モデルがそろっていれば、実機 E2E テストも実行できます。

```bash
mise run test:integration
```

[次へ: 各コンポーネントの基本 →](02-components.md)
