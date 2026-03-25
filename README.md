# Event Graph Generation

動画から構造化されたイベントグラフを自動生成する深層学習フレームワーク。

SAM 3 によるオブジェクト検出・追跡、Qwen VL による合成アノテーション生成、DETR 風 Event Decoder によるセット予測を組み合わせ、動画中の「誰が・何を・どこから・どこへ」をグラフ構造として出力します。

## 主な機能 / ユースケース

- **構造化出力**: テキスト生成ではなく、Hungarian Matching によるセット予測でイベントグラフを直接出力
- **End-to-End パイプライン**: 動画入力から EventGraph JSON 出力まで一貫した処理フロー
- **合成データ生成**: SAM 3 のトラッキング結果と Qwen VL のアノテーションを自動アライメントし、学習データを構築
- **軽量モデル**: Event Decoder は約 10-15M パラメータで高速推論が可能
- **スライディングウィンドウ推論**: 長時間動画をクリップ分割 + NMS 重複排除で処理
- **マルチバックエンド VLM**: transformers / VLLM / VLLM Server (OpenAI 互換 API) から選択可能

### ユースケース

- 製造・組立作業の動画からの作業イベント自動抽出
- 室内行動の構造的記録（誰が何をどこに置いたか等）
- 動画理解のための構造化アノテーション自動生成

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 言語 | Python >= 3.13 |
| 深層学習 | PyTorch >= 2.10, torchvision >= 0.25 |
| オブジェクト追跡 | SAM 3 (Segment Anything Model 3) |
| VLM アノテーション | Qwen 3.5 (transformers / VLLM) |
| モデルアーキテクチャ | DETR 風セット予測 (Event Decoder) |
| 量子化 | bitsandbytes >= 0.48 |
| 実験管理 | Weights & Biases (wandb) |
| スキーマバリデーション | Pydantic >= 2.0 |
| 最適化 | SciPy (Hungarian Matching) |
| 画像処理 | OpenCV >= 4.8 |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |
| ビルドシステム | Hatchling |

## モデル情報

### SAM 3 (Segment Anything Model 3)

- **用途**: 動画中のオブジェクト検出・追跡・セグメンテーション
- **モデルサイズ**: Large（デフォルト設定）
- **パッケージ**: `sam3 >= 0.1.3`（PyPI）
- **埋め込み次元**: 256

### Qwen 3.5 VL (Vision-Language Model)

合成アノテーション生成に使用。バックエンドに応じて異なるモデルを選択可能。

| バックエンド | モデル | 設定ファイル |
|---|---|---|
| transformers | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `configs/vlm.yaml` |
| VLLM | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `configs/vlm_vllm.yaml` |
| VLLM Server | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `configs/vlm_vllm_server.yaml` |

### Event Decoder

- **アーキテクチャ**: DETR 風セット予測モデル（Temporal Encoder → Context Encoder → Event Decoder → 7 予測ヘッド）
- **パラメータ数**: 約 10-15M
- **学習方式**: Hungarian Matching による最適割当（非自己回帰）

| 予測ヘッド | 出力 | 説明 |
|---|---|---|
| interaction | (M, 1) | 有効なイベントか（BCE） |
| action | (M, A) | アクション分類（13クラス） |
| agent_ptr | (M, K) | 行為者オブジェクトへのポインタ |
| target_ptr | (M, K) | 対象オブジェクトへのポインタ |
| source_ptr | (M, K+1) | 取り出し元へのポインタ（任意） |
| dest_ptr | (M, K+1) | 格納先へのポインタ（任意） |
| frame | (M, T) | イベント発生フレーム |

## 使用データ

### 入力データ

- 動画ファイル（MP4 形式）
- `data/` ディレクトリは共有ストレージ (`/share/koi_hackathon/data`) へのシンボリックリンク

### 中間データ

| データ | パス | 形式 | 説明 |
|---|---|---|---|
| リサイズ済み動画 | `data/resized/` | MP4 | SAM 3 入力解像度に合わせたもの |
| SAM 3 出力 | `data/sam3_outputs/` | PT | オブジェクト追跡結果（bbox, mask, embedding） |
| VLM アノテーション | `data/annotations/` | JSON | Qwen VL による合成アノテーション |
| タイムスタンプ付きアノテーション | `data/annotations_enriched/` | JSON | 絶対時刻・メタデータ付き |
| アライメント済み学習データ | `data/aligned/samples/` | PT | 特徴量 + GT イベント |

### 学習データ形式 (`*.pt`)

```python
{
    "object_embeddings": Tensor (N, D_emb),    # SAM 3 オブジェクト埋め込み
    "object_temporal": Tensor (N, T, D_geo),   # 幾何特徴 (bbox, centroid, area, velocity 等)
    "pairwise": Tensor (N, N, T, D_pair),      # ペアワイズ特徴 (IoU, 距離, 包含関係等)
    "gt_events": list[dict]                     # 正解イベント
}
```

## セットアップ

### 前提環境

| 項目 | 要件 |
|---|---|
| OS | Linux（Ubuntu 推奨） |
| Python | >= 3.13 |
| GPU | CUDA 対応 GPU（VRAM 16GB 以上推奨） |
| パッケージマネージャ | [uv](https://docs.astral.sh/uv/) |
| CUDA | PyTorch 対応バージョン |

VLM バックエンドごとの追加 GPU 要件:

| バックエンド | 推奨 VRAM |
|---|---|
| transformers (Qwen3.5-9B) | 24GB+ (単一 GPU) |
| VLLM (Qwen3.5-35B-A3B) | 4x GPU, `tensor_parallel_size=4` |
| VLLM Server | 上記と同等（別プロセスで起動） |

### インストール

```bash
# uv がない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係インストール
uv sync
```

**オプショナル依存（必要に応じて別途インストール）:**

```bash
# VLLM バックエンドを使用する場合
pip install vllm>=0.11.0

# VLLM Server バックエンド (OpenAI 互換 API) を使用する場合
pip install openai
```

> **注意**: `transformers` は PyPI リリースではなく GitHub main ブランチからインストールされます（`pyproject.toml` で指定済み）。`vllm` は transformers の git main と競合する場合があるため、`pyproject.toml` には含めていません。

### 環境変数

| 変数名 | 必須 | 説明 |
|---|---|---|
| `WANDB_API_KEY` | 任意 | Weights & Biases の API キー（実験追跡を有効にする場合） |
| `HF_TOKEN` | 任意 | Hugging Face トークン（gated model をダウンロードする場合） |
| `CUDA_VISIBLE_DEVICES` | 任意 | 使用する GPU の指定 |

WandB を使用しない場合は、設定ファイルで `wandb.enabled: false` に設定すれば環境変数は不要です。

## 実行方法

### パイプライン全体の流れ

データセット構築から学習・推論まで、`scripts/` 配下のスクリプトを番号順に実行します。

```
1_resize_videos.py → 2_run_sam3_tracking.py → 3_generate_annotations.py
    → 4_build_dataset.py → 5_train.py → 6_evaluate.py → 7_run_inference.py
```

### 1. 学習データ構築

```bash
# Step 1: 動画リサイズ（SAM 3 解像度に合わせる）
uv run python scripts/1_resize_videos.py \
  --input-dir data/mp4 \
  --output-dir data/resized

# Step 2: SAM 3 によるオブジェクト追跡
uv run python scripts/2_run_sam3_tracking.py \
  --config configs/sam3.yaml \
  --video-dir data/mp4

# Step 3: VLM による合成アノテーション生成
uv run python scripts/3_generate_annotations.py \
  --config configs/vlm.yaml \
  --video-dir data/raw/

# Step 4: アライメント + データセット構築
uv run python scripts/4_build_dataset.py \
  --config configs/training.yaml
```

**共通フラグ:**

| フラグ | 説明 |
|---|---|
| `--config` | YAML 設定ファイルの指定 |
| `--override` | 実験オーバーライド設定（深いマージ） |
| `--resume` | 処理済みスキップ（バッチ処理ステージ） |
| `--shard-id` / `--num-shards` | マルチ GPU 並列化（script 2） |

### 2. 学習

```bash
# 基本
uv run python scripts/5_train.py --config configs/training.yaml

# 実験オーバーライド付き
uv run python scripts/5_train.py \
  --config configs/training.yaml \
  --override configs/experiment/event_decoder_v1.yaml

# 学習再開
uv run python scripts/5_train.py \
  --config configs/training.yaml \
  --resume data/checkpoints/epoch_0050.pt
```

### 3. 評価

```bash
uv run python scripts/6_evaluate.py --config configs/training.yaml
```

### 4. 推論（動画 → EventGraph）

```bash
uv run python scripts/7_run_inference.py \
  --config configs/inference.yaml \
  --video data/raw/video.mp4 \
  --checkpoint data/checkpoints/best.pt \
  --output output/event_graph.json
```

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | `configs/inference.yaml` | 推論設定ファイル |
| `--video` | （必須） | 入力動画パス |
| `--checkpoint` | （必須） | 学習済みチェックポイント |
| `--output` | `output/event_graph.json` | 出力先パス |
| `--confidence-threshold` | `0.5` | イベント検出の信頼度閾値 |
| `--actions-config` | `configs/actions.yaml` | アクション語彙定義 |

### 出力例（EventGraph JSON）

```json
{
  "video_id": "assembly_001",
  "nodes": [
    {
      "track_id": 0,
      "category": "person",
      "first_seen_frame": 0,
      "last_seen_frame": 120
    }
  ],
  "edges": [
    {
      "event_id": "evt_0000",
      "agent_track_id": 0,
      "action": "pick_up",
      "target_track_id": 1,
      "source_track_id": 2,
      "frame": 15,
      "confidence": 0.92
    }
  ]
}
```

### テスト

```bash
uv run pytest                         # 全テスト
uv run pytest tests/test_config.py    # 特定ファイル
uv run pytest -k test_accuracy        # パターン指定
```

## エントリーポイント

| スクリプト | 役割 |
|---|---|
| `scripts/1_resize_videos.py` | 動画リサイズ（SAM 3 入力解像度） |
| `scripts/2_run_sam3_tracking.py` | SAM 3 によるオブジェクト追跡 |
| `scripts/3_generate_annotations.py` | VLM による合成アノテーション生成 |
| `scripts/4_build_dataset.py` | アライメント + 学習データセット構築 |
| `scripts/5_train.py` | Event Decoder の学習 |
| `scripts/6_evaluate.py` | モデル評価 |
| `scripts/7_run_inference.py` | 推論（動画 → EventGraph JSON） |
| `scripts/enrich_timestamps.py` | 既存アノテーションへのタイムスタンプ付与 |
| `scripts/benchmark_vram.py` | VRAM 使用量ベンチマーク |

シェルラッパー:
- `scripts/local/` — ローカル実行用
- `scripts/slurm/` — SLURM クラスタ用（sbatch）

## ディレクトリ構成

```
event-graph-generation/
├── src/event_graph_generation/    # メインパッケージ
│   ├── schemas/                   #   データ構造定義 (ObjectNode, EventEdge, EventGraph, VLMAnnotation)
│   ├── annotation/                #   VLM 合成アノテーション (vlm_annotator, prompts, alignment, validator)
│   ├── tracking/                  #   SAM 3 ラッパー + 時系列/ペアワイズ特徴抽出
│   ├── data/                      #   データセット (EventGraphDataset), バッチ collation, フレームサンプリング
│   ├── models/                    #   Event Decoder, 予測ヘッド (MLP), 損失関数 (EventGraphLoss)
│   ├── training/                  #   学習ループ (AMP, early stopping, WandB), optimizer/scheduler
│   ├── evaluation/                #   評価器, メトリクスレジストリ
│   ├── inference/                 #   推論パイプライン (sliding window + NMS), 後処理
│   ├── config.py                  #   Dataclass ベース YAML 設定管理
│   └── utils/                     #   シード固定, ロギング, チェックポイント I/O, モーション検出
├── scripts/                       # 実行スクリプト（番号順パイプライン）
│   ├── 1_resize_videos.py
│   ├── 2_run_sam3_tracking.py
│   ├── 3_generate_annotations.py
│   ├── 4_build_dataset.py
│   ├── 5_train.py
│   ├── 6_evaluate.py
│   ├── 7_run_inference.py
│   ├── enrich_timestamps.py
│   ├── benchmark_vram.py
│   ├── local/                     #   ローカル実行用シェルラッパー
│   └── slurm/                     #   SLURM 用シェルラッパー
├── configs/                       # YAML 設定ファイル
│   ├── default.yaml               #   ベースデフォルト設定
│   ├── training.yaml              #   学習ハイパーパラメータ
│   ├── inference.yaml             #   推論パイプライン設定
│   ├── vlm.yaml                   #   VLM 設定 (transformers)
│   ├── vlm_vllm.yaml             #   VLM 設定 (VLLM)
│   ├── vlm_vllm_server.yaml      #   VLM 設定 (VLLM Server)
│   ├── sam3.yaml                  #   SAM 3 トラッキング設定
│   ├── sam3_kitchen.yaml          #   ドメイン別 SAM 3 設定
│   ├── sam3_desk.yaml
│   ├── sam3_room.yaml
│   ├── actions.yaml               #   アクション語彙定義（13クラス）
│   ├── vocab.yaml                 #   オブジェクトカテゴリ・属性語彙
│   └── experiment/                #   実験ごとのオーバーライド
├── tests/                         # テストコード
├── data/                          # データディレクトリ（共有ストレージへのシンボリックリンク）
└── pyproject.toml                 # プロジェクト定義・依存関係
```

## 制約・注意事項

### 計算資源

- SAM 3 トラッキングには CUDA 対応 GPU が必須
- VLM アノテーション生成は使用モデルに応じて大量の VRAM を要求（Qwen3.5-35B-A3B は 4GPU 推奨）
- 学習・推論は GPU 上での実行を前提とし、CPU のみの環境ではサポート外
- テストはすべて CPU 上で小さい次元の合成データで実行可能

### 外部依存

- `transformers` は PyPI リリースではなく **GitHub main ブランチ**からインストールされる（Qwen 3.5 サポートのため）
- `sam3` はオプショナル依存。未インストールでも SAM 3 以外の機能（学習、評価等）は動作する
- `vllm` はオプショナル。`transformers` の git main と競合する可能性があるため `pyproject.toml` には未宣言
- `openai` はオプショナル。VLLM Server バックエンド使用時のみ必要

### データ

- `data/` ディレクトリは共有ストレージ (`/share/koi_hackathon/data`) へのシンボリックリンクを想定
- 動画データは別途用意が必要（リポジトリには含まれない）

### その他

- すべての Python 実行は `uv run` 経由で行うこと（仮想環境の手動 activate は不要）
- 設定は dataclass ベースの YAML で管理。`--override` フラグにより深いマージが可能

## ライセンス

TBD
