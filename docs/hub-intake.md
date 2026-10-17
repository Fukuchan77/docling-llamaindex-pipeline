# 位置づけとハブへの取り込み（hub intake）

- **作成日**: 2026-10-17
- **宛先**: `vaz-agentic-ai-next`（統合ハブ。ADR-0003）
- **関係リポジトリ**: `pydantic-ai-sandbox`（パターン検証。ハブ取り込みは同リポジトリの `docs/hub-intake-2026-10.md` と同じ運用）

散文は日本語、識別子・パス・コードは英語。

## 1. 役割

本リポジトリは **RAG に特化した検証リポジトリ** である。ドキュメント解析（Docling）を伴う RAG を、
ハブへ持ち込む前にここで検証する。ハブへは統合しない（独立維持）。

| RAG の種類 | 担当 | 条件 |
| :--- | :--- | :--- |
| 軽量 RAG（テキスト中心。TS が pgvector へ書き込む） | ハブ `packages/rag` | ハブの single-writer 原則（pgvector への書き込みは TS のみ） |
| 文書解析が必要な RAG（PDF の OCR・表・レイアウト、出典のページ番号と BBox） | 本リポジトリのパイプライン | 解析〜取り込み〜検索を本リポジトリの API で完結させる |

使い分けの境界（どの入力から本リポジトリ側へ回すか）は、ライブ検証後に本書へ追記する。

## 2. モデル対応の計画（ハブに合わせる）

ハブは Anthropic / OpenAI / Ollama を `AI_PROVIDER` で切り替える。本リポジトリも同じ軸に揃える。

| 項目 | 現状 | 計画 |
| :--- | :--- | :--- |
| 回答生成 LLM | Ollama（既定）/ OpenAI | **Anthropic を追加**（`llama-index-llms-anthropic`、extra `anthropic`） |
| 埋め込み | Ollama（既定）/ OpenAI / IBM | 変更なし。Anthropic に埋め込み API は無いため、Anthropic LLM 選択時も埋め込みは Ollama か OpenAI |
| モデル ID | `config.py` の既定値のみ | 同じ規則を維持。ハブの許可リストとの突き合わせは実装時に行う |
| モデル方針 | IBM Granite 既定、中国製モデル禁止（`tests/unit/test_model_policy.py`） | 既定は維持。Anthropic / OpenAI は選択式の追加とし、方針テストの対象に含める |

## 3. Evidence ledger

status は `proposed` / `verified` / `landed` / `rejected` のいずれか。完了・不採用の行も残す。

| ID | 変更 | status | 根拠・次の条件 |
| :--- | :--- | :--- | :--- |
| R1 | Anthropic LLM プロバイダの追加 | landed | 2026-10-17 に実装（`LLM_PROVIDER=anthropic`、extra `anthropic`）。実 API での疎通は未確認 |
| R2 | ライブ検証（実 Ollama / Docling モデル。`mise run test:integration`） | proposed | 未実行。PR #1 は hermetic な単体テストのみ green |
| R3 | BBox 付き出典（`grounding.py`）をハブ `services/agent` の `/parse` locator と突き合わせ | proposed | ハブ側の locator は `{source, locator, ordinal, text}`。R2 の後に差分を確認 |
| R4 | OCR エンジン固定（`DOCLING_OCR_ENGINE`） | proposed | Docling 自動選択が RapidOCR（PaddleOCR 由来）を選びうるため。ハブ `services/agent` の Docling 設定と比較 |
| R5 | Ollama の `num_ctx` 上限 | proposed | 本リポジトリ `bb4b531` で対応済み。ハブ側 Ollama 経路へ必要か未確認 |

取り込みはファイルのコピーではなく、ハブ側での再実装とする。
