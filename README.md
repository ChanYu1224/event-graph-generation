# Event Graph Generation

動画から構造化されたイベントグラフを自動生成する深層学習フレームワーク。

V-JEPA による映像特徴抽出、Qwen VL による合成アノテーション生成、DETR 風 Event Decoder によるセット予測を組み合わせ、動画中の「誰が・何を・どこから・どこへ」をグラフ構造として出力します。

## 主な機能 / ユースケース

- **構造化出力**: テキスト生成ではなく、Hungarian Matching によるセット予測でイベントグラフを直接出力
- **End-to-End パイプライン**: 動画入力から EventGraph JSON 出力まで一貫した処理フロー
- **合成データ生成**: V-JEPA の映像特徴と Qwen VL のアノテーションを組み合わせて学習データを構築
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
| 映像特徴抽出 | V-JEPA 2.1 (PyTorch Hub) |
| VLM アノテーション | Qwen 3.5 (transformers / VLLM) |
| モデルアーキテクチャ | Object Pooling + DETR 風セット予測 (Event Decoder) |
| 量子化 | bitsandbytes >= 0.48 |
| 実験管理 | Weights & Biases (wandb) |
| スキーマバリデーション | Pydantic >= 2.0 |
| 最適化 | SciPy (Hungarian Matching) |
| 画像処理 | OpenCV >= 4.8 |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |
| ビルドシステム | Hatchling |

## モデル情報

### V-JEPA 2.1 (Video Joint Embedding Predictive Architecture)

- **用途**: 動画からの時空間特徴トークン抽出
- **モデルバリエーション**: ViT-B / ViT-L / ViT-G / ViT-gigantic
- **取得先**: PyTorch Hub (`facebookresearch/vjepa`)
- **入力解像度**: 384px, 16 frames/clip
- **出力**: 時空間トークン (B, S, D) — S = temporal_tokens x spatial_tokens

### Qwen 3.5 VL (Vision-Language Model)

合成アノテーション生成に使用。バックエンドに応じて異なるモデルを選択可能。

| バックエンド | モデル | 設定ファイル |
|---|---|---|
| transformers | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `configs/vlm.yaml` |
| VLLM | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `configs/vlm_vllm.yaml` |
| VLLM Server | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `configs/vlm_vllm_server.yaml` |

### Event Decoder

- **アーキテクチャ**: V-JEPA トークン → Object Pooling (Slot Attention) → DETR 風 Event Decoder → 7 予測ヘッド
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
| リサイズ済み動画 | `data/resized/` | MP4 | V-JEPA 入力解像度 (384px) に合わせたもの |
| V-JEPA 特徴 | `data/vjepa_features/` | PT | 時空間トークン (S, D) |
| VLM アノテーション | `data/annotations/` | JSON | Qwen VL による合成アノテーション |
| タイムスタンプ付きアノテーション | `data/annotations_enriched/` | JSON | 絶対時刻・メタデータ付き |
| 学習データセット | `data/vjepa_aligned/` | PT + split txt | V-JEPA 特徴 + GT イベント |

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

データセット構築から学習・評価まで、`scripts/` 配下のスクリプトを番号順に実行します。

```
0_resize_videos.py → 1_extract_features.py → 2_generate_annotations.py
    → 3_build_dataset.py → 4_train.py → 5_evaluate.py
```

### 1. 学習データ構築

```bash
# Step 0: 動画リサイズ（V-JEPA 入力解像度 384px に合わせる）
uv run python scripts/0_resize_videos.py \
  --input-dir data/mp4 \
  --output-dir data/resized

# Step 1: V-JEPA 特徴抽出
uv run python scripts/1_extract_features.py \
  --config configs/vjepa.yaml \
  --video-dir data/resized \
  --output-dir data/vjepa_features

# Step 2: VLM による合成アノテーション生成
uv run python scripts/2_generate_annotations.py \
  --config configs/vlm.yaml \
  --video-dir data/resized/ \
  --resume

# Step 3: データセット構築
uv run python scripts/3_build_dataset.py \
  --features-dir data/vjepa_features \
  --output-dir data/vjepa_aligned \
  --annotations-dir data/annotations \
  --vocab configs/vocab.yaml \
  --actions configs/actions.yaml
```

**共通フラグ:**

| フラグ | 説明 |
|---|---|
| `--config` | YAML 設定ファイルの指定 |
| `--override` | 実験オーバーライド設定（深いマージ） |
| `--resume` | 処理済みスキップ（バッチ処理ステージ） |

### 2. 学習

```bash
# 基本
uv run python scripts/4_train.py --config configs/vjepa_training.yaml

# 実験オーバーライド付き
uv run python scripts/4_train.py \
  --config configs/vjepa_training.yaml \
  --override configs/experiment/vjepa_vitb.yaml

# 学習再開
uv run python scripts/4_train.py \
  --config configs/vjepa_training.yaml \
  --resume data/checkpoints/epoch_0050.pt
```

### 3. 評価

```bash
uv run python scripts/5_evaluate.py --config configs/vjepa_training.yaml
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
| `scripts/0_resize_videos.py` | 動画リサイズ（V-JEPA 入力解像度 384px） |
| `scripts/1_extract_features.py` | V-JEPA 特徴抽出 |
| `scripts/2_generate_annotations.py` | VLM による合成アノテーション生成 |
| `scripts/3_build_dataset.py` | V-JEPA 特徴 + アノテーション → 学習データセット構築 |
| `scripts/4_train.py` | V-JEPA Pipeline の学習 |
| `scripts/5_evaluate.py` | モデル評価 |
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
│   ├── tracking/                  #   トラッキング・特徴抽出
│   ├── data/                      #   データセット (VJEPAEventDataset), バッチ collation, フレームサンプリング
│   ├── models/                    #   VJEPAPipeline, ObjectPooling, EventDecoder, 損失関数
│   ├── training/                  #   学習ループ (AMP, early stopping, WandB), optimizer/scheduler
│   ├── evaluation/                #   評価器, メトリクスレジストリ
│   ├── inference/                 #   推論パイプライン (sliding window + NMS), 後処理
│   ├── config.py                  #   Dataclass ベース YAML 設定管理
│   └── utils/                     #   シード固定, ロギング, チェックポイント I/O, モーション検出
├── scripts/                       # 実行スクリプト（番号順パイプライン）
│   ├── 0_resize_videos.py
│   ├── 1_extract_features.py
│   ├── 2_generate_annotations.py
│   ├── 3_build_dataset.py
│   ├── 4_train.py
│   ├── 5_evaluate.py
│   ├── enrich_timestamps.py
│   ├── benchmark_vram.py
│   ├── local/                     #   ローカル実行用シェルラッパー
│   └── slurm/                     #   SLURM 用シェルラッパー
├── configs/                       # YAML 設定ファイル
│   ├── default.yaml               #   ベースデフォルト設定
│   ├── vjepa.yaml                 #   V-JEPA 特徴抽出設定
│   ├── vjepa_training.yaml        #   V-JEPA Pipeline 学習ハイパーパラメータ
│   ├── vlm.yaml                   #   VLM 設定 (transformers)
│   ├── vlm_vllm.yaml             #   VLM 設定 (VLLM)
│   ├── vlm_vllm_server.yaml      #   VLM 設定 (VLLM Server)
│   ├── actions.yaml               #   アクション語彙定義（13クラス）
│   ├── vocab.yaml                 #   オブジェクトカテゴリ・属性語彙
│   └── experiment/                #   実験ごとのオーバーライド (vjepa_vitb, vjepa_vitg 等)
├── tests/                         # テストコード
├── data/                          # データディレクトリ（共有ストレージへのシンボリックリンク）
└── pyproject.toml                 # プロジェクト定義・依存関係
```

## 制約・注意事項

### 計算資源

- V-JEPA 特徴抽出・学習・推論には CUDA 対応 GPU が必須
- VLM アノテーション生成は使用モデルに応じて大量の VRAM を要求（Qwen3.5-35B-A3B は 4GPU 推奨）
- テストはすべて CPU 上で小さい次元の合成データで実行可能

### 外部依存

- `transformers` は PyPI リリースではなく **GitHub main ブランチ**からインストールされる（Qwen 3.5 サポートのため）
- V-JEPA モデルは PyTorch Hub から自動ダウンロード（初回実行時にインターネット接続が必要）
- `vllm` はオプショナル。`transformers` の git main と競合する可能性があるため `pyproject.toml` には未宣言
- `openai` はオプショナル。VLLM Server バックエンド使用時のみ必要

### データ

- `data/` ディレクトリは共有ストレージ (`/share/koi_hackathon/data`) へのシンボリックリンクを想定
- 動画データは別途用意が必要（リポジトリには含まれない）

### その他

- すべての Python 実行は `uv run` 経由で行うこと（仮想環境の手動 activate は不要）
- 設定は dataclass ベースの YAML で管理。`--override` フラグにより深いマージが可能

## ライセンス

MIT License
