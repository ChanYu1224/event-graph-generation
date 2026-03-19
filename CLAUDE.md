# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

動画から構造化されたイベントグラフを生成する深層学習フレームワーク。SAM 3でオブジェクトの検出・追跡を行い、Qwen VLで合成アノテーションを生成し、軽量なEvent Decoder（DETR風セット予測モデル）でイベントを構造化JSONとして出力する。テキスト生成ではなく、Hungarian Matchingによるセット予測でイベントグラフを直接出力する設計。

## Commands

**すべてのPython実行は `uv run` 経由で行うこと。** 仮想環境の手動activateは不要。

```bash
# 依存関係インストール
uv sync

# テスト
uv run pytest                                # 全テスト
uv run pytest tests/test_config.py           # 特定ファイル
uv run pytest -k test_accuracy               # パターン指定

# 学習（Event Decoder）
uv run python scripts/train.py --config configs/training.yaml
uv run python scripts/train.py --config configs/training.yaml --override configs/experiment/event_decoder_v1.yaml
uv run python scripts/train.py --config configs/training.yaml --resume data/checkpoints/epoch_0050.pt

# 推論（End-to-End: 動画 → EventGraph JSON）
uv run python scripts/run_inference.py --config configs/inference.yaml --video data/raw/video.mp4

# データセット構築パイプライン（順に実行）
uv run python scripts/run_sam3_tracking.py --config configs/sam3.yaml --video-dir data/raw/
uv run python scripts/generate_annotations.py --config configs/vlm.yaml --video-dir data/raw/
uv run python scripts/build_dataset.py --config configs/training.yaml

# VRAMベンチマーク
uv run python scripts/benchmark_vram.py
```

## Architecture

### Data Flow (End-to-End Pipeline)

```
Video → FrameSampler (1fps) → frames
  ├→ SAM3Tracker → tracking results (objects, embeddings, masks, bboxes per frame)
  │    └→ FeatureExtractor → ObjectFeatures + PairwiseFeatures (.pt)
  └→ VLMAnnotator (Qwen 3.5) → VLMAnnotation (objects, events)
       └→ Aligner (Hungarian matching, bbox IoU) → VLM obj ↔ SAM3 track mapping
            └→ Aligned training samples (.pt in data/aligned/samples/)

Training: EventGraphDataset loads .pt → EventDecoder learns event prediction
Inference: SAM3 → FeatureExtractor → EventDecoder → postprocess → EventGraph JSON
```

### Source Modules

```
src/event_graph_generation/
├── schemas/           # ObjectNode, EventEdge, EventGraph dataclasses; VLMAnnotation (Pydantic)
├── annotation/        # VLM-based synthetic annotation: vlm_annotator, prompts, alignment, validator
├── tracking/          # SAM3 wrapper (sam3_tracker) + temporal/pairwise feature extraction
├── data/              # EventGraphDataset (.pt loader), EventBatch collator, FrameSampler (OpenCV)
├── models/            # EventDecoder (DETR-style), 7 prediction heads (MLP), EventGraphLoss
├── training/          # Trainer (AMP, early stopping, WandB), optimizer/scheduler factories
├── evaluation/        # Evaluator, metric registry (event_detection_map, action_accuracy, etc.)
├── inference/         # InferencePipeline (sliding window + NMS dedup), postprocessing
├── config.py          # Dataclass-based YAML config: Config.from_yaml(), merge(), to_yaml()
└── utils/             # seed_everything, setup_logging, save/load_checkpoint
```

### Event Decoder Model (DETR-style)

入力: object_embeddings `(B,K,D_emb)` + object_temporal `(B,K,T,D_geo)` + pairwise `(B,K,K,T,D_pair)` + object_mask `(B,K)`

処理: Temporal Encoder → Context Encoder (self-attention) → Event Decoder (cross-attention, learnable event queries M個)

出力 (7つの予測ヘッド):
- `interaction` (M,1): 有効なイベントか (BCE)
- `action` (M,A): アクション分類 (CE) — 13クラス (`configs/actions.yaml`)
- `agent_ptr`, `target_ptr` (M,K): オブジェクトスロットへのポインタ (CE)
- `source_ptr`, `dest_ptr` (M,K+1): コンテナへのポインタ (+1は"none") — アクション依存で任意
- `frame` (M,T): イベント発生フレーム分類 (CE)

学習時はHungarian Matchingで予測↔GTの最適割当を行い、未マッチの予測はno_eventとして扱う。

## Design Patterns

- **Factory**: `build_model()`, `build_optimizer()`, `build_scheduler()`, `get_metric()`
- **Dataclass Config**: 全設定がdataclass。`Config.from_yaml()` → `merge()` で実験オーバーライド
- **Metric Registry**: 文字列名でメトリクス検索 (`METRIC_REGISTRY`)
- **Set Prediction**: 固定M個のevent queries、Hungarian Matching、非自己回帰
- **Sliding Window**: 推論時にclip_length/clip_strideでオーバーラップ処理 + NMS重複排除

## Configuration

- `configs/training.yaml` — Event Decoder学習ハイパーパラメータ（ベース設定）
- `configs/inference.yaml` — 推論パイプライン設定（SAM3, clip, NMS）
- `configs/vlm.yaml` — Qwen 3.5 VLMモデル設定
- `configs/sam3.yaml` — SAM3トラッキング設定 + concept_prompts
- `configs/actions.yaml` — アクション語彙定義（13クラス、各アクションにsource/destination要否フラグ）
- `configs/experiment/` — 実験ごとのオーバーライド（`--override`で深いマージ）

## Key Dependencies

- **uv** でパッケージ管理、Python >= 3.13
- torch >= 2.10, transformers (git main), accelerate
- pydantic >= 2.0（VLM出力のスキーマバリデーション）
- scipy（Hungarian Matching: `linear_sum_assignment`）
- opencv-python（フレーム抽出）
- wandb（実験追跡）、bitsandbytes（量子化）

## Data Format

学習データ（`data/aligned/samples/*.pt`）の各サンプル:
```python
{
    "object_embeddings": Tensor (N, D_emb),    # SAM3オブジェクト埋め込み
    "object_temporal": Tensor (N, T, D_geo),   # 幾何特徴（bbox, centroid, area, velocity等）
    "pairwise": Tensor (N, N, T, D_pair),      # ペアワイズ特徴（IoU, 距離, 包含関係等）
    "gt_events": list[dict]                     # GTイベント（action, agent, target, frame等）
}
```
`EventGraphDataset`がmax_objects=Kにパディングし、`event_collate_fn`でバッチ化。
