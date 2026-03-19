# アーキテクチャ概要

本ドキュメントでは、Event Graph Generation フレームワーク全体のアーキテクチャを説明します。

## 設計思想

本フレームワークの核となる設計判断:

1. **Event Decoder の出力はテキストではなくグラフ構造体** — 自己回帰デコードは行わず、DETR 風のセット予測で一括出力する
2. **SAM 3 の Vision Encoder は凍結** — 学習対象は Event Decoder（約 10-15M パラメータ）のみ
3. **VLM を教師データ生成に使用** — 推論時には VLM は不要で、軽量な Event Decoder のみで動作

## パイプライン全体図

```
┌─────────────────────────────────────────────────────────┐
│                   データ生成パイプライン                      │
│                                                         │
│  動画ファイル                                             │
│    │                                                    │
│    ├─ FrameSampler (1fps)                               │
│    │    │                                               │
│    │    ├─ SAM 3 Tracker                                │
│    │    │    └─ FrameTrackingResult (per frame)          │
│    │    │         └─ FeatureExtractor                   │
│    │    │              ├─ ObjectFeatures (per track)     │
│    │    │              └─ PairwiseFeatures (per pair)    │
│    │    │                                               │
│    │    └─ VLM Annotator (Qwen 3.5)                     │
│    │         └─ VLMAnnotation (objects, events)          │
│    │              └─ AnnotationValidator                 │
│    │                                                    │
│    └─ Aligner (Hungarian Matching)                      │
│         └─ AlignmentResult (VLM obj ↔ SAM 3 track)     │
│              └─ 学習サンプル (.pt)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     学習パイプライン                        │
│                                                         │
│  EventGraphDataset (.pt ファイル読み込み)                   │
│    └─ EventBatch (event_collate_fn でバッチ化)            │
│         └─ EventDecoder (forward pass)                  │
│              └─ EventGraphLoss (Hungarian Matching)      │
│                   └─ Trainer (AMP, WandB, EarlyStopping) │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     推論パイプライン                        │
│                                                         │
│  動画ファイル                                             │
│    └─ InferencePipeline                                 │
│         ├─ FrameSampler → フレーム列                      │
│         ├─ SAM3Tracker → 追跡結果                         │
│         ├─ FeatureExtractor → 特徴量                     │
│         ├─ EventDecoder (sliding window) → predictions  │
│         ├─ 重複排除 (NMS-like)                           │
│         └─ EventGraph JSON 出力                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## モジュール構成

| モジュール | パッケージ | 役割 |
|---|---|---|
| [スキーマ](./schemas.md) | `schemas/` | データ構造の定義（EventGraph, VLMAnnotation） |
| [トラッキング](./tracking.md) | `tracking/` | SAM 3 によるオブジェクト追跡と特徴抽出 |
| [アノテーション](./annotation.md) | `annotation/` | VLM による合成アノテーション生成とバリデーション |
| [データ](./data.md) | `data/` | データセット、バッチ処理、フレームサンプリング |
| [モデル](./models.md) | `models/` | Event Decoder モデルと損失関数 |
| [学習](./training.md) | `training/` | 学習ループ、optimizer、scheduler |
| [評価](./evaluation.md) | `evaluation/` | 評価器とメトリクスレジストリ |
| [推論](./inference.md) | `inference/` | End-to-End 推論パイプライン |
| [設定](./config.md) | `config.py` | Dataclass ベースの YAML 設定管理 |

## 主要な設計パターン

### Factory パターン

モデル、optimizer、scheduler、メトリクスはすべて文字列名で構築可能:

```python
model = build_model(config.model)            # "event_decoder" → EventDecoder
optimizer = build_optimizer(model, config)     # "adamw" → AdamW
scheduler = build_scheduler(optimizer, config) # "cosine_warmup" → CosineAnnealing
metric_fn = get_metric("graph_f1")            # "graph_f1" → graph_f1()
```

### Dataclass Config

全設定が Python dataclass として型付けされ、YAML との相互変換が可能:

```python
config = Config.from_yaml("configs/training.yaml")
config = config.merge("configs/experiment/v1.yaml")  # deep merge
config.to_yaml("output/resolved_config.yaml")
```

### DETR 風セット予測

- 固定数の learnable event queries（デフォルト M=20）
- Hungarian Matching で予測↔GT の最適割当
- 未マッチの予測は "no_event" として interaction loss のみ適用

## データの流れ

### テンソル形状の変遷

```
SAM 3 出力（1フレーム）:
  TrackedObject: mask (H,W), bbox [x1,y1,x2,y2], embedding (256,)

FeatureExtractor 出力:
  ObjectFeatures: embedding (D_emb,), bbox_seq (T,4), centroid_seq (T,2), ...
  PairwiseFeatures: iou_seq (T,1), distance_seq (T,1), ...

学習サンプル (.pt):
  object_embeddings: (N, D_emb=256)
  object_temporal:   (N, T=16, D_geo=12)
  pairwise:          (N, N, T=16, D_pair=7)

EventGraphDataset (パディング後):
  object_embeddings: (K=30, D_emb)
  object_temporal:   (K, T, D_geo)
  pairwise:          (K, K, T, D_pair)
  object_mask:       (K,) bool

EventBatch (バッチ化後):
  object_embeddings: (B, K, D_emb)
  object_temporal:   (B, K, T, D_geo)
  pairwise:          (B, K, K, T, D_pair)
  object_mask:       (B, K)

EventDecoder 出力:
  interaction: (B, M=20, 1)
  action:      (B, M, A=13)
  agent_ptr:   (B, M, K=30)
  target_ptr:  (B, M, K)
  source_ptr:  (B, M, K+1)
  dest_ptr:    (B, M, K+1)
  frame:       (B, M, T=16)
```

### 特徴量の構成

**ObjectFeatures の D_geo=12 の内訳:**

| index | 名前 | 次元 | 説明 |
|---|---|---|---|
| 0-3 | bbox | 4 | 正規化 cx, cy, w, h |
| 4-5 | centroid | 2 | 正規化 cx, cy |
| 6 | area | 1 | w × h |
| 7 | presence | 1 | 0/1 indicator |
| 8-9 | delta_centroid | 2 | フレーム間重心差分 |
| 10 | delta_area | 1 | フレーム間面積差分 |
| 11 | velocity | 1 | 重心差分の L2 ノルム |

**PairwiseFeatures の D_pair=7 の内訳:**

| index | 名前 | 次元 | 説明 |
|---|---|---|---|
| 0 | iou | 1 | bbox の IoU |
| 1 | distance | 1 | 正規化重心間距離 |
| 2 | containment_ij | 1 | i の j への包含率 |
| 3 | containment_ji | 1 | j の i への包含率 |
| 4-5 | relative_position | 2 | 相対位置 (dx, dy) |
| 6 | symmetric_iou | 1 | IoU（対称特徴） |
