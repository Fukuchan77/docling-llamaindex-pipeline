# 2. 各コンポーネントの基本

API サーバを組み立てる前に、Docling と LlamaIndex × Ollama をそれぞれ単体で動かして確認します。
ここで扱う処理は、[`examples/`](../../examples) に実行可能なスクリプトとして収録しています。
スクリプトは CI で lint と型チェックを受けています。

[← セットアップ](01-setup.md) ・ [ガイド目次](README.md) ・ [次へ: API の使い方 →](03-api-usage.md)

---

## 2.1 Docling: CLI で文書を変換する

> 公式: [Docling Usage](https://docling-project.github.io/docling/usage/) ・ [CLI リファレンス](https://docling-project.github.io/docling/reference/cli/) ・ [対応フォーマット](https://docling-project.github.io/docling/usage/supported_formats/)

```bash
# 標準パイプライン（レイアウト解析 + TableFormer）で Markdown に変換
uv run docling report.pdf --ocr-engine tesseract --ocr-lang jpn,eng --to md --output ./output

# JSON（DoclingDocument）。ページ番号や座標などのプロバナンスを含む
uv run docling report.pdf --ocr-engine tesseract --ocr-lang jpn,eng --to json --output ./output
```

> **Warning:** `--ocr-engine` を省略すると、Docling は RapidOCR（PaddleOCR 由来のモデル）を自動選択し、モデルをダウンロードします。
> 本プロジェクトの方針（PaddleOCR 由来モデルを使わない）に反するため、CLI では必ず `--ocr-engine tesseract` を付けてください。

| 対応形式 | 必要なモデル |
| :--- | :--- |
| PDF | レイアウトモデル + TableFormer（[1.5](01-setup.md#15-docling-のモデルとチャンク用トークナイザ)） |
| DOCX / PPTX / XLSX / HTML / CSV / XML（JATS・USPTO） | 不要（文書の構造をそのまま読み取る） |

本 API が受け付ける形式は上表の 7 種です（仕様 §4.2）。

## 2.2 Docling: Granite-Docling-258M（VLM）で変換する

> 公式: [IBM Granite-Docling 発表](https://www.ibm.com/jp-ja/new/announcements/granite-docling-end-to-end-document-conversion) ・ [IBM Granite Docs: Docling](https://www.ibm.com/granite/docs/models/docling) ・ [Hugging Face: granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)

Granite-Docling は 258M パラメータの視覚言語モデルです。ページ画像から DocTags（構造・座標・数式・表を保持するマークアップ）を直接生成します。
数式やコード、複雑な表や多段組を含む技術文書に向いています。

```bash
# VLM プリセットは granite_docling を明示的に指定する（他社モデルのプリセットもあるため）
uv run docling paper.pdf --pipeline vlm --vlm-model granite_docling --to md --output ./output
```

Python SDK での使い方は [`examples/01_docling_basic.py`](../../examples/01_docling_basic.py) を参照してください。

```bash
uv run python examples/01_docling_basic.py report.pdf          # 標準パイプライン（Tesseract OCR）
uv run python examples/01_docling_basic.py paper.pdf --vlm     # Granite-Docling-258M
```

要点は次のとおりです。

```python
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TesseractCliOcrOptions, VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# 標準: OCR エンジンを明示（自動選択だと PaddleOCR 由来モデルが選ばれ得る）
standard = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(
        ocr_options=TesseractCliOcrOptions(lang=["iso:jpn", "iso:eng"]),
    )),
})

# VLM: Granite-Docling-258M を明示
granite = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=VlmPipelineOptions(vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS),
    ),
})

doc = standard.convert("report.pdf").document     # DoclingDocument
print(doc.export_to_markdown()[:400])
```

| 項目 | 内容 |
| :--- | :--- |
| 追加パッケージ（transformers 版） | 追加なしで動作する。`docling[vlm]` は他の VLM 用の依存も含む |
| 追加パッケージ（Apple Silicon） | `uv sync --extra mlx`（`mlx-vlm` を Apple Silicon にだけ導入。`mise run setup` でも自動で入る）。`docling[vlm]` も同じ `mlx-vlm` を含む |
| Apple Silicon | `vlm_model_specs.GRANITEDOCLING_MLX`。`examples/01_docling_basic.py --vlm` は `mlx-vlm` があれば自動で選ぶ |
| 実測（Apple Silicon、`ISWP2026_Chap1.pdf` の 2 ページ） | MLX 約 8 秒/ページ（2 ページ 16.8 秒）、transformers 約 135 秒/ページ（2 ページ 289 秒）。約 17 倍の差 |
| 本 API での扱い | 現在は標準パイプラインを使用（VLM パイプラインの API 組み込みは今後の拡張） |

## 2.3 LlamaIndex × Ollama: Granite モデルの疎通確認

> 公式: [Ollama LLM 連携](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/) ・ [Ollama Embedding 連携](https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding/) ・ [Using LLMs](https://developers.llamaindex.ai/python/framework/understanding/using_llms/)

```bash
uv run python examples/02_ollama_basic.py
# embedding: granite-embedding:278m -> 768 dims
# [granite4.2:latest]
# ...（日本語の回答）
```

[`examples/02_ollama_basic.py`](../../examples/02_ollama_basic.py) は、API と同じ設定（`OLLAMA_*`）からモデル名を読みます。
ここで成功すれば、API サーバからも同じモデルに到達できます。

```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed = OllamaEmbedding(model_name="granite-embedding:278m", base_url="http://localhost:11434")
llm = Ollama(model="granite4.2:latest", base_url="http://localhost:11434",
             request_timeout=120.0, context_window=8192, temperature=0.1)

len(embed.get_text_embedding("ローカル RAG"))   # 768
llm.complete("RAG の利点を一文で").text
```

> **Note:** LlamaIndex の公式チュートリアルは `Settings.llm` / `Settings.embed_model`（グローバル設定）を使います。
> 本 API は埋め込みのホットスワップが実行中のリクエストに影響しないよう、グローバル設定は変更しません。
> モデルを `VectorStoreIndex` やクエリエンジンへ明示的に渡しています（[`rag_service.py`](../../src/docling_llamaindex_pipeline/rag_service.py)）。

## 2.4 Docling × LlamaIndex: 構造を保ったチャンク分割

> 公式: [Docling Reader / Node Parser デモ](https://developers.llamaindex.ai/python/examples/data_connectors/doclingreaderdemo/) ・ [Docling チャンク分割](https://docling-project.github.io/docling/concepts/chunking/) ・ [HybridChunker の例](https://docling-project.github.io/docling/examples/hybrid_chunking/) ・ [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/)

本 API の取り込み処理（[`parsing.py`](../../src/docling_llamaindex_pipeline/parsing.py)）は、次の 2 つを組み合わせています。

| 部品 | 役割 |
| :--- | :--- |
| `DoclingReader(export_type=JSON)` | 文書の論理構造・ページ番号・座標を失わずに LlamaIndex へ渡す |
| `DoclingNodeParser(chunker=HybridChunker(...))` | 章・節・表の境界に沿って分割し、さらに埋め込みモデルのトークン上限内に収める |

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

# OCR を明示する。省略すると既定のコンバータが RapidOCR（PaddleOCR 由来）を選ぶ
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(
        ocr_options=TesseractCliOcrOptions(lang=["iso:jpn", "iso:eng"]),
    )),
})
reader = DoclingReader(export_type=DoclingReader.ExportType.JSON, doc_converter=converter)
tokenizer = HuggingFaceTokenizer.from_pretrained(
    model_name="ibm-granite/granite-embedding-278m-multilingual",  # granite-embedding:278m と同じ語彙
    max_tokens=512,                                                 # 同モデルのコンテキスト上限
)
parser = DoclingNodeParser(chunker=HybridChunker(tokenizer=tokenizer, merge_peers=True))

nodes = parser.get_nodes_from_documents(reader.load_data("report.pdf"))
nodes[0].metadata["headings"]                 # 例: ['第3章 財務ハイライト', '3.2 連結損益計算書']
nodes[0].metadata["doc_items"][0]["prov"]     # 例: [{'page_no': 14, 'bbox': {...}, ...}]
```

**なぜトークン上限を合わせるのか**

`granite-embedding:278m` の入力上限は 512 トークンです。これを超えたチャンクは、Ollama 側で黙って切り詰められます。
その結果、長い節の後半は検索に一度もヒットしなくなります。
`CHUNKER=hybrid`（既定）は、トークナイザでトークン数を数えながら分割するので、この問題を防げます。

| `CHUNKER` | 動作 | 使いどころ |
| :--- | :--- | :--- |
| `hybrid`（既定） | 構造境界で分割し、かつ `CHUNK_MAX_TOKENS` 以内に収める | 通常はこちら |
| `hierarchical` | 文書要素ごとに 1 チャンク、トークン上限なし | トークナイザを用意できない環境での一時的な回避策 |

埋め込みモデルを変える場合は、`CHUNK_TOKENIZER` と `CHUNK_MAX_TOKENS` もそのモデルに合わせて変更してください。

**メタデータの扱い**

- `doc_items`・`origin` などのメタデータは、`DoclingNodeParser` によって埋め込み対象から除外されます。
- 一方、`headings`（見出し階層）は埋め込みにも LLM のプロンプトにも含まれ、検索精度の向上に寄与します。
- 本 API はこれらに加え、`file_name` を各ノードに付与します。

[次へ: API の使い方 →](03-api-usage.md)
