# docling-llamaindex-pipeline

Docling × LlamaIndex によるマルチフォーマット対応の取り込み & グラウンディング付き RAG API。
PDF / DOCX / PPTX / XLSX / HTML / CSV / XML を 1 つのエンドポイントで解析し、回答と一緒に
引用元のページ番号とバウンディングボックスを返します。既定構成（Docling + Ollama）では
外部 SaaS への送信は一切ありません。

- **利用ガイド（セットアップ〜基本的な使い方、公式リンク付き）: [`docs/guide/`](docs/guide/README.md)**
- 仕様・基本設計: [`docs/SPEC-RAG-2026-001.md`](docs/SPEC-RAG-2026-001.md)
- 位置づけとハブ取り込み（RAG 特化の検証リポジトリ）: [`docs/hub-intake.md`](docs/hub-intake.md)

既定モデルは IBM Granite に統一しています。中国製の LLM・埋め込み・OCR モデルは使用しません（`tests/unit/test_model_policy.py` で検査）。

| 役割 | モデル |
| :--- | :--- |
| 回答生成 | `granite4.2:latest`（Ollama） |
| 埋め込み | `granite-embedding:278m`（Ollama、多言語・768 次元・512 トークン） |
| 文書解析 | Docling（IBM レイアウトモデル + TableFormer）/ 任意で Granite-Docling-258M |
| OCR | Tesseract（既定）/ EasyOCR |

## クイックスタート

```bash
mise install            # Python 3.14 / uv（mise.toml で固定）
mise run setup          # uv sync --all-extras --all-groups + pre-commit install
cp .env.example .env    # 必要に応じて編集

ollama pull granite4.2:latest && ollama pull granite-embedding:278m
sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn   # PDF の OCR 用（macOS: brew install tesseract tesseract-lang）
mise run dev            # http://127.0.0.1:8000/docs
```

mise がない場合: `uv sync --all-extras --all-groups` のあと `uv run docling-llamaindex-pipeline`。

Ollama のみで使う場合、オプション extra は不要です（`uv sync`）。OpenAI は `--extra openai`、
IBM watsonx.ai は `--extra ibm` で追加します。

> Docling は PyPI から CUDA 版 torch を取り込むため、初回 `uv sync` は約 6 GB / 数分かかります。

## API

| メソッド | パス | 説明 |
| :--- | :--- | :--- |
| `POST` | `/api/v1/ingest` | ファイル（multipart `file`）を Docling で解析しインデックスに登録（201） |
| `POST` | `/api/v1/query` | `{"query", "top_k"=4, "export_grounding"=true}` → 回答 + ソース + BBox |
| `POST` | `/api/v1/config/embedding` | `{"provider", "model_name"?}` で埋め込みをホットスワップ（インデックスは初期化） |
| `GET` | `/health` | ドキュメント数 / ノード数 / 現行プロバイダ |

```bash
curl -F file=@quarterly_report.pdf localhost:8000/api/v1/ingest
curl -H 'Content-Type: application/json' \
     -d '{"query": "2025年度の営業利益率の目標は?", "top_k": 3}' localhost:8000/api/v1/query
```

### エラー

| 状況 | ステータス |
| :--- | :--- |
| 未対応拡張子 / 空ファイル / 抽出可能テキストなし / 未インデックスでの検索 / 不明プロバイダ・認証情報不足 | 400 |
| `ADMIN_API_KEY` 設定時の `X-Admin-Token` 不一致 | 401 |
| `MAX_UPLOAD_SIZE_MB` 超過 | 413 |
| Docling パース失敗（破損・暗号化 PDF 等）/ その他の内部エラー | 500 |
| 埋め込み・LLM サーバに接続できない | 503 |
| `QUERY_TIMEOUT_SECONDS` 超過 | 504 |

レスポンスは FastAPI 標準の `{"detail": "..."}` 形式です。

### バウンディングボックス座標

Docling の PDF 座標は **左下原点（`BOTTOMLEFT`）・ポイント単位** です。Canvas / CSS（左上原点）
で描画する場合は `top = page_height - t`, `left = l`, `width = r - l`, `height = t - b`。
HTML / DOCX / CSV などページレイアウトを持たない形式では `bounding_boxes` は空になります。

## 設定

すべて環境変数（または `.env`）で指定します。一覧と既定値は [`.env.example`](.env.example)、
型と検証は [`config.py`](src/docling_llamaindex_pipeline/config.py) を参照。

- 起動時に検証し、誤設定は即座に起動失敗（選択プロバイダの認証情報不足など）。
- `APP_ENV=production` では `CORS_ALLOW_ORIGINS` の `*` を禁止し、`ADMIN_API_KEY` を必須化。
- エアギャップ運用: `docling-tools models download layout tableformer` で取得したモデルを `DOCLING_ARTIFACTS_PATH` に指定し、チャンク用トークナイザを HF キャッシュに入れて `HF_HUB_OFFLINE=1`（[手順](docs/guide/01-setup.md#15-docling-のモデルとチャンク用トークナイザ)）。
- チャンク分割は `HybridChunker` + `granite-embedding-278m-multilingual` トークナイザで 512 トークン以内に制御（`CHUNKER` / `CHUNK_MAX_TOKENS`）。

## 開発

品質ゲートはすべて `mise` タスク経由です（CI も同じタスクを呼びます）。

| タスク | 内容 |
| :--- | :--- |
| `mise run check` | lint + format + typecheck + カバレッジ付きテスト（合格基準） |
| `mise run lint` / `format` / `format:fix` | Ruff |
| `mise run typecheck` | Pyright strict + ty（src と tests） |
| `mise run test` / `cov` | ユニットテスト（ネットワーク・Docling モデル不要）/ カバレッジ 98% 以上 |
| `mise run test:integration` | 実 Docling + 実 Ollama の E2E（`RUN_INTEGRATION_OLLAMA=1`） |
| `mise run audit` | uv.lock に対する pip-audit |
| `mise run dev` / `serve` | 開発サーバ（ホットリロード）/ 本番相当起動 |

ユニットテストは `HF_HUB_OFFLINE=1` で完全にオフラインです。HTML / CSV は実 Docling で解析し、
PDF 相当のプロバナンス（ページ・BBox）は `DoclingDocument` を直接組み立てて実
`DoclingNodeParser` に通すことで、レイアウトモデルなしで検証しています。

### 構成

```
src/docling_llamaindex_pipeline/
├── app.py          # create_app(): ミドルウェア・lifespan・ルータの組み立てのみ
├── config.py       # Settings（唯一のモデル ID 既定値の置き場）
├── providers.py    # 埋め込み / LLM ファクトリ（OpenAI・IBM は遅延 import）
├── parsing.py      # DoclingReader(JSON) + DoclingNodeParser(HybridChunker)、OCR エンジン固定
├── uploads.py      # ファイル名サニタイズ・一時保存・finally での削除
├── grounding.py    # doc_items[].prov[] → BoundingBox
├── rag_service.py  # VectorStoreIndex・ロック・ホットスワップ
├── errors.py       # ドメイン例外 → HTTP ステータス
├── schemas.py      # API 入出力モデル
└── api/            # deps（AppContainer, 管理トークン）と各ルート
examples/           # ガイドから参照する実行可能サンプル（Docling 単体 / Ollama 疎通 / API クライアント）
```

### サンプル実装からの主な変更点

- アップロード名をパスに直接使わない（パストラバーサル対策）。リクエストごとの `0700` 一時ディレクトリに保存し必ず削除。
- `DocumentConverter` をプロセスで 1 つ再利用（リクエストごとのモデル再ロードを回避）し、Docling 実行はワーカースレッド + 同時実行数制限。
- 埋め込み・検索は非同期 API（`ainsert_nodes` / `aquery`）。グローバル `llama_index.core.Settings` は変更せず、モデルを明示的に渡す。
- ホットスワップは新モデルの構築に成功してからインデックスを切り替え、取り込みとはロックで直列化。`model_name` 指定も反映。
- LLM プロバイダを `LLM_PROVIDER` として埋め込みプロバイダから独立。
- 413 / 503 / 504 の追加、CORS の既定無効化、管理 API のトークン保護（任意、本番必須）。

## 今後の拡張（仕様 §8）

ベクトル DB 永続化（Qdrant / pgvector 等）、BM25 とのハイブリッド検索（RRF）、リランカー、SSE ストリーミング。
`RagService` がインデックスを一手に持っているため、ストレージ差し替えはここに閉じます。
