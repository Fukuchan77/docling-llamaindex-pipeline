# 3. API の使い方

文書を取り込み、質問し、回答の根拠（ページ・座標）を受け取るまでの流れを説明します。
エンドポイントの仕様は、サーバ起動後に `http://127.0.0.1:8000/docs`（Swagger UI）で確認できます。

[← 各コンポーネントの基本](02-components.md) ・ [ガイド目次](README.md)

---

## 3.1 サーバを起動する

> 公式: [FastAPI](https://fastapi.tiangolo.com/) ・ [Uvicorn: Settings](https://www.uvicorn.org/settings/) ・ [mise Tasks](https://mise.jdx.dev/tasks/)

```bash
mise run dev     # ホットリロード付き。.env を読み込む
mise run serve   # 本番相当。API_HOST / API_PORT で待受先を指定（コンテナでは API_HOST=0.0.0.0）
```

mise を使わない場合は `uv run docling-llamaindex-pipeline` で起動できます。

> **Note:** 起動時には、チャンク分割用トークナイザを読み込みます（[1.5](01-setup.md#15-docling-のモデルとチャンク用トークナイザ)）。
> 取得できない場合は `Could not load chunk tokenizer ...` というエラーで起動が止まります。

## 3.2 文書を取り込む — `POST /api/v1/ingest`

> 公式: [FastAPI: Request Files](https://fastapi.tiangolo.com/tutorial/request-files/) ・ [Docling 対応フォーマット](https://docling-project.github.io/docling/usage/supported_formats/)

```bash
curl -F file=@quarterly_report.pdf http://127.0.0.1:8000/api/v1/ingest
```

```json
{
  "status": "SUCCESS",
  "file_name": "quarterly_report.pdf",
  "total_nodes_created": 18,
  "active_embedding_provider": "ollama",
  "sample_nodes": [
    {"node_id": "7f8b…", "text": "3.2 連結損益計算書 当第3四半期累計期間における売上高は…",
     "headings": ["第3章 財務ハイライト", "3.2 連結損益計算書"]}
  ]
}
```

| 項目 | 内容 |
| :--- | :--- |
| 対応拡張子 | `.pdf` `.docx` `.pptx` `.xlsx` `.html` `.csv` `.xml` |
| 上限サイズ | `MAX_UPLOAD_SIZE_MB`（既定 50 MB） |
| 一時ファイル | アップロードはリクエストごとの非公開ディレクトリに保存し、解析後に必ず削除 |
| 所要時間（CPU） | PDF 1 ページあたり 1.5〜3 秒程度（仕様 §6.2） |
| 所要時間（初回 PDF） | Docling モデルの読み込み分だけ余計にかかる |

## 3.3 質問する — `POST /api/v1/query`

> 公式: [LlamaIndex: VectorStoreIndex](https://developers.llamaindex.ai/python/framework/module_guides/indexing/vector_store_index/) ・ [Ollama LLM 連携](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/)

```bash
curl -H 'Content-Type: application/json' \
     -d '{"query": "2025年度の営業利益率の目標値とその根拠となる施策は?", "top_k": 3}' \
     http://127.0.0.1:8000/api/v1/query
```

| フィールド | 既定値 | 説明 |
| :--- | :--- | :--- |
| `query` | （必須） | 質問文（1〜8192 文字） |
| `top_k` | `4` | 取得するチャンク数（1〜20） |
| `export_grounding` | `true` | `false` にすると `bounding_boxes` を省略 |

```json
{
  "query": "2025年度の営業利益率の目標値とその根拠となる施策は?",
  "answer": "2025年度の営業利益率目標は12.5%です。…",
  "provider": "ollama",
  "sources": [
    {
      "node_id": "7f8b…",
      "text_snippet": "中期経営計画目標: 2025年度営業利益率 12.5% を必達目標とする。…",
      "score": 0.8924,
      "headings": ["中期経営計画", "重点財務目標"],
      "file_name": "mid_term_plan.pdf",
      "bounding_boxes": [
        {"page_no": 14, "l": 72.0, "t": 520.4, "r": 523.5, "b": 480.1, "coord_origin": "BOTTOMLEFT"}
      ]
    }
  ]
}
```

| 項目 | 内容 |
| :--- | :--- |
| 回答の生成 | `granite4.2:latest` が、取得したチャンクの本文と見出しをもとに生成する |
| `sources` の並び | 類似度（`score`）の高い順 |
| `bounding_boxes` | PDF のみ。HTML / DOCX / CSV などページレイアウトを持たない形式では空 |

## 3.4 根拠の座標を画面に描く

> 公式: [DoclingDocument（プロバナンス）](https://docling-project.github.io/docling/concepts/docling_document/)

Docling の PDF 座標は **左下原点（`BOTTOMLEFT`）・ポイント単位** です。
ブラウザの Canvas や CSS は左上原点なので、ページの高さ `H` を使って次のように変換します（仕様 §5.1）。

```text
left = l      top = H - t      width = r - l      height = t - b
```

| 用紙 | `H` |
| :--- | :--- |
| A4 | 841.89 pt |
| US Letter | 792 pt |

[`examples/03_api_client.py`](../../examples/03_api_client.py) は、取り込み → 質問 → 座標変換までを一通り実行します。

```bash
uv run python examples/03_api_client.py mid_term_plan.pdf "営業利益率の目標は?" --page-height 841.89
# Q: 営業利益率の目標は?
# A: …
# [1] mid_term_plan.pdf  score=0.892  section=中期経営計画 > 重点財務目標
#     page 14: {'left': 72.0, 'top': 321.49, 'width': 451.5, 'height': 40.3}
```

## 3.5 埋め込みプロバイダの切り替え — `POST /api/v1/config/embedding`

> 公式: [Ollama Embedding 連携](https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding/) ・ [Granite Embedding（Hugging Face）](https://huggingface.co/ibm-granite)

```bash
# 同じ Ollama 上で、別の Granite 埋め込みモデル（英語専用の軽量版）へ切り替える例
curl -H 'Content-Type: application/json' -H "X-Admin-Token: $ADMIN_API_KEY" \
     -d '{"provider": "ollama", "model_name": "granite-embedding:30m"}' \
     http://127.0.0.1:8000/api/v1/config/embedding
```

| 項目 | 内容 |
| :--- | :--- |
| `provider` | `ollama` / `ibm`（watsonx.ai、`--extra ibm`）/ `openai`（`--extra openai`） |
| 切り替え後のインデックス | 初期化される。ベクトルの次元や意味空間がモデルごとに異なるため。文書の再取り込みが必要 |
| 切り替えに失敗した場合 | 400 を返し、元のインデックスはそのまま残る（認証情報不足、未インストールの extra など） |
| `X-Admin-Token` | `ADMIN_API_KEY` を設定しているときだけ必要 |
| チャンク分割の設定 | 埋め込みモデルの上限が変わる場合は、`CHUNK_TOKENIZER` / `CHUNK_MAX_TOKENS` も合わせて再起動する |

## 3.6 ヘルスチェック — `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

このエンドポイントはインメモリの状態だけを返し、Ollama には問い合わせません。
そのため、モデルサーバの再起動中も 200 を返します。
ロードバランサや Kubernetes の readiness / liveness probe に使えます。

## 3.7 エラーと対処

レスポンスは `{"detail": "..."}` 形式です。

| ステータス | 主な原因 | 対処 |
| :--- | :--- | :--- |
| 400 | 未対応拡張子 / 空ファイル / テキストを抽出できない（OCR なしのスキャン PDF など） / 未取り込みでの質問 / 不明なプロバイダ | 入力と設定を確認する |
| 401 | `X-Admin-Token` がない、または一致しない | `ADMIN_API_KEY` と同じ値を送る |
| 413 | `MAX_UPLOAD_SIZE_MB` 超過 | 上限を上げるか、ファイルを分割する |
| 500 | Docling の解析失敗（破損・暗号化 PDF、Tesseract 未インストール など） | サーバログを確認する |
| 503 | Ollama に接続できない | `ollama serve` と `OLLAMA_BASE_URL` を確認し、`ollama list` にモデルがあるか確認する |
| 504 | `QUERY_TIMEOUT_SECONDS` 超過 | CPU 推論時はタイムアウトを延ばすか、`top_k` を減らす |

### よくある質問

**回答に推論過程（thinking）が混ざる**
Granite 4.2 は思考モードを持つモデルです（[IBM Granite 4.2](https://www.ibm.com/granite/docs/models/granite4-2)）。
Ollama 側の思考モードの扱いは [Ollama ドキュメント](https://docs.ollama.com/) を参照してください。

**PDF だけ 500 になる**
次の 2 点を確認してください。
- Tesseract がインストールされているか（`tesseract --list-langs`）。
- Docling モデルを取得できているか（[1.4](01-setup.md#14-ocrtesseract) / [1.5](01-setup.md#15-docling-のモデルとチャンク用トークナイザ)）。

**取り込んだ文書が再起動で消える**
現在のインデックスはインメモリです。
永続化（Qdrant / pgvector など）は仕様 §8 の拡張計画に含まれています（[LlamaIndex: Using Vector Stores](https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/)）。
