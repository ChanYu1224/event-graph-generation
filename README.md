# Event Graph Generation

動画から構造化されたイベントグラフを自動生成するフレームワーク。

SAM 3 によるオブジェクト検出・追跡、Qwen VL による合成アノテーション生成、DETR 風 Event Decoder によるセット予測を組み合わせ、動画中の「誰が・何を・どこから・どこへ」をグラフ構造として出力します。

## 特徴

- **構造化出力**: テキスト生成ではなく、Hungarian Matching によるセット予測でイベントグラフを直接出力
- **End-to-End パイプライン**: 動画入力から EventGraph JSON 出力まで一貫した推論パイプライン
- **合成データ生成**: SAM 3 のトラッキング結果と Qwen VL のアノテーションを自動アライメントし、学習データを構築
- **軽量モデル**: Event Decoder は約 10-15M パラメータで高速推論が可能

## アーキテクチャ

```
動画ファイル
  │
  ├─ FrameSampler (1fps) ─→ フレーム列
  │     │
  │     ├─ SAM 3 Tracker ─→ オブジェクト追跡結果 (bbox, mask, embedding)
  │     │     └─ FeatureExtractor ─→ 時系列特徴 + ペアワイズ特徴
  │     │
  │     └─ VLM Annotator (Qwen 3.5) ─→ 合成アノテーション (objects, events)
  │           └─ Aligner (Hungarian matching) ─→ VLM ↔ SAM 3 対応付け
  │                 └─ 学習データ (.pt)
  │
  └─ 推論パイプライン
        SAM 3 → FeatureExtractor → Event Decoder → EventGraph JSON
```

### EventGraph の構造

EventGraph はノード（オブジェクト）とエッジ（イベント）から構成されるグラフです。

```json
{
  "video_id": "assembly_001",
  "nodes": [
    {
      "track_id": 0,
      "category": "person",
      "first_seen_frame": 0,
      "last_seen_frame": 120
    },
    {
      "track_id": 1,
      "category": "wrench",
      "first_seen_frame": 10,
      "last_seen_frame": 95
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

### Event Decoder

DETR 風のセット予測モデル。固定数の learnable event queries を用いて、クロスアテンションでオブジェクト表現からイベントを予測します。

| ヘッド | 出力 | 説明 |
|---|---|---|
| interaction | (M, 1) | 有効なイベントか（BCE） |
| action | (M, A) | アクション分類（13クラス） |
| agent_ptr | (M, K) | 行為者オブジェクトへのポインタ |
| target_ptr | (M, K) | 対象オブジェクトへのポインタ |
| source_ptr | (M, K+1) | 取り出し元へのポインタ（任意） |
| dest_ptr | (M, K+1) | 格納先へのポインタ（任意） |
| frame | (M, T) | イベント発生フレーム |

## セットアップ

### 前提条件

- Python >= 3.13
- CUDA 対応 GPU
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

### インストール

```bash
uv sync
```

## 使い方

### 推論（動画 → EventGraph）

```bash
uv run python scripts/run_inference.py \
  --config configs/inference.yaml \
  --video data/raw/video.mp4 \
  --checkpoint data/checkpoints/best.pt \
  --output output/event_graph.json
```

オプション:

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | `configs/inference.yaml` | 推論設定ファイル |
| `--video` | （必須） | 入力動画パス |
| `--checkpoint` | （必須） | 学習済み Event Decoder チェックポイント |
| `--output` | `output/event_graph.json` | 出力先パス |
| `--concept-prompts` | 設定ファイルから | SAM 3 のオブジェクト検出プロンプト |
| `--confidence-threshold` | `0.5` | イベント検出の信頼度閾値 |
| `--actions-config` | `configs/actions.yaml` | アクション語彙定義 |

### 学習データ構築

学習データの構築は以下の 3 ステップで行います。

```bash
# 1. SAM 3 によるオブジェクト追跡
uv run python scripts/run_sam3_tracking.py \
  --config configs/sam3.yaml \
  --video-dir data/raw/

# 2. VLM による合成アノテーション生成
uv run python scripts/generate_annotations.py \
  --config configs/vlm.yaml \
  --video-dir data/raw/

# 3. アライメント + データセット構築
uv run python scripts/build_dataset.py \
  --config configs/training.yaml
```

生成される学習データ（`data/aligned/samples/*.pt`）:

```python
{
    "object_embeddings": Tensor (N, D_emb),    # SAM 3 オブジェクト埋め込み
    "object_temporal": Tensor (N, T, D_geo),   # 幾何特徴 (bbox, centroid, area, velocity 等)
    "pairwise": Tensor (N, N, T, D_pair),      # ペアワイズ特徴 (IoU, 距離, 包含関係等)
    "gt_events": list[dict]                     # 正解イベント
}
```

### 学習

```bash
# 基本
uv run python scripts/train.py --config configs/training.yaml

# 実験オーバーライド付き
uv run python scripts/train.py \
  --config configs/training.yaml \
  --override configs/experiment/event_decoder_v1.yaml

# 学習再開
uv run python scripts/train.py \
  --config configs/training.yaml \
  --resume data/checkpoints/epoch_0050.pt
```

### テスト

```bash
uv run pytest                         # 全テスト
uv run pytest tests/test_config.py    # 特定ファイル
uv run pytest -k test_accuracy        # パターン指定
```

## 設定ファイル

| ファイル | 説明 |
|---|---|
| `configs/training.yaml` | 学習ハイパーパラメータ（lr, scheduler, loss weights 等） |
| `configs/inference.yaml` | 推論設定（SAM 3, clip 分割, NMS 重複排除） |
| `configs/vlm.yaml` | Qwen 3.5 VLM モデル設定 |
| `configs/sam3.yaml` | SAM 3 トラッキング設定 + concept prompts |
| `configs/actions.yaml` | アクション語彙定義（13クラス） |
| `configs/experiment/` | 実験ごとのオーバーライド |

設定は dataclass ベースで、`--override` フラグにより深いマージが可能です。

```bash
# configs/training.yaml をベースに、experiment 設定を上書き
uv run python scripts/train.py \
  --config configs/training.yaml \
  --override configs/experiment/event_decoder_v1.yaml
```

## アクション語彙

`configs/actions.yaml` で定義される 13 種類のアクション:

| アクション | 説明 | source | destination |
|---|---|---|---|
| take_out | 容器/収納から取り出す | 要 | - |
| put_in | 容器/収納に入れる | - | 要 |
| place_on | 面の上に置く | - | 要 |
| pick_up | 面の上から持ち上げる | 要 | - |
| hand_over | 人から人へ渡す | 要 | 要 |
| open | 容器/引き出しを開ける | - | - |
| close | 容器/引き出しを閉める | - | - |
| use | 工具/道具を使用する | - | - |
| move | 場所 A から場所 B へ移動する | 要 | 要 |
| attach | 物体を取り付ける | - | 要 |
| detach | 物体を取り外す | 要 | - |
| inspect | 物体を視認/確認する | - | - |
| no_event | イベントなし（negative class） | - | - |

## プロジェクト構成

```
src/event_graph_generation/
├── schemas/           # データ構造 (ObjectNode, EventEdge, EventGraph, VLMAnnotation)
├── annotation/        # VLM 合成アノテーション (vlm_annotator, prompts, alignment, validator)
├── tracking/          # SAM 3 ラッパー + 時系列/ペアワイズ特徴抽出
├── data/              # データセット (EventGraphDataset), バッチ collation, フレームサンプリング
├── models/            # Event Decoder, 予測ヘッド (MLP), 損失関数 (Hungarian matching)
├── training/          # 学習ループ (AMP, early stopping, WandB), optimizer/scheduler
├── evaluation/        # 評価器, メトリクスレジストリ
├── inference/         # 推論パイプライン (sliding window + NMS), 後処理
├── config.py          # Dataclass ベース YAML 設定管理
└── utils/             # シード固定, ロギング, チェックポイント I/O

scripts/
├── train.py                  # 学習エントリポイント
├── run_inference.py          # 推論エントリポイント
├── run_sam3_tracking.py      # SAM 3 トラッキング実行
├── generate_annotations.py   # VLM アノテーション生成
├── build_dataset.py          # アライメント + データセット構築
├── benchmark_vram.py         # VRAM ベンチマーク
└── *.sh                      # SLURM バッチジョブスクリプト

configs/
├── training.yaml             # 学習設定
├── inference.yaml            # 推論設定
├── vlm.yaml                  # VLM 設定
├── sam3.yaml                 # SAM 3 設定
├── actions.yaml              # アクション語彙
└── experiment/               # 実験オーバーライド
```

## 主要な依存関係

- **PyTorch** >= 2.10 / **torchvision** >= 0.25
- **transformers** (git main branch) — Qwen VL モデル
- **accelerate** >= 1.7 — 分散学習サポート
- **pydantic** >= 2.0 — VLM 出力のスキーマバリデーション
- **scipy** >= 1.11 — Hungarian Matching (`linear_sum_assignment`)
- **opencv-python** >= 4.8 — フレーム抽出
- **wandb** >= 0.25 — 実験追跡
- **bitsandbytes** >= 0.48 — 量子化

## ライセンス

TBD
