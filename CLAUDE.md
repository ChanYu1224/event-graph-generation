# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**重要**: コードに変更を加えた際は、このファイルの関連セクション（Architecture, Configuration, Data Format等）も合わせて更新すること。ドキュメントとコードの乖離を防ぐ。

## Project Overview

動画から構造化されたイベントグラフを生成する深層学習フレームワーク。SAM 3でオブジェクトの検出・追跡を行い、Qwen VLで合成アノテーションを生成し、軽量なEvent Decoder（DETR風セット予測モデル）でイベントを構造化JSONとして出力する。テキスト生成ではなく、Hungarian Matchingによるセット予測でイベントグラフを直接出力する設計。

## Commands

**すべてのPython実行は `uv run` 経由で行うこと。** 仮想環境の手動activateは不要。

```bash
uv sync                    # 依存関係インストール
uv run pytest              # 全テスト（-k でパターン指定可）
```

データセット構築パイプラインは `scripts/{1..7}_*.py` の番号順に実行。共通フラグ:
- `--config` / `--override`: YAML設定の指定と実験オーバーライド（深いマージ）
- `--resume`: 処理済みスキップ（バッチ処理ステージ）
- `--shard-id` / `--num-shards`: マルチGPU並列化（script 2）

```bash
uv run python scripts/2_run_sam3_tracking.py --config configs/sam3_kitchen.yaml --video-dir data/mp4
uv run python scripts/5_train.py --config configs/training.yaml --override configs/experiment/event_decoder_v1.yaml
uv run python scripts/5_train.py --config configs/training.yaml --resume data/checkpoints/epoch_0050.pt
```

タイムスタンプ付与（既存アノテーションのバックフィル）:
```bash
uv run python scripts/enrich_timestamps.py \
  --annotations-dir data/annotations --output-dir data/annotations_enriched \
  --video-dir data/resized/room --target-fps 1.0 --clip-length 16 --clip-stride 8
```

- `data/` は共有ストレージ (`/share/koi_hackathon/data`) へのシンボリックリンク
- `scripts/local/` と `scripts/slurm/` に各ステージのシェルラッパーあり

## Coding Conventions

- `from __future__ import annotations` を全モジュール先頭に記載
- 型ヒント: `X | None` を使用（`Optional[X]` は使わない）
- PEP 257 docstring: 全public関数に `Args:` / `Returns:` セクション付き
- ロギング: `logger = logging.getLogger(__name__)` を全モジュールで使用、`%s` 形式を推奨
- インポート: src内は相対インポート（`from .heads import PredictionHead`）、テスト・スクリプトは絶対インポート
- データ構造: 内部データは `@dataclass`、VLM外部出力バリデーション (`schemas/vlm_output.py`) のみ Pydantic `BaseModel`
- 命名: `snake_case`（関数/変数）、`CamelCase`（クラス）、`_private`（内部メソッド）、`UPPERCASE`（定数）

## Testing Conventions

- テストファイル: `tests/test_<module_name>.py`
- テストデータ: `_make_*()` ヘルパー関数（`**kwargs` でデフォルトオーバーライド）
- テストクラス: 関連テストをクラスでグループ化（`TestBboxConversion` 等）
- Fixture: `@pytest.fixture` はインフラ用（`extractor` 等）、データ生成は `_make_*()` で
- 浮動小数点比較: `pytest.approx()`
- GPU不要: 全テストCPU上で小さい次元のsyntheticデータで実行
- conftest.py なし: fixture はテストファイルごとにローカル定義

## Architecture

### Data Flow (End-to-End Pipeline)

#### SAM3パイプライン（レガシー）
```
Video → FrameSampler (1fps) → frames
  ├→ SAM3Tracker → tracking results (objects, embeddings, masks, bboxes per frame)
  │    └→ FeatureExtractor → ObjectFeatures + PairwiseFeatures (.pt)
  └→ VLMAnnotator (Qwen 3.5, motion filter で静止クリップ除外)
       │  backends: transformers / vllm / vllm-server (OpenAI API)
       ├→ VLMAnnotation (objects, events)
       │    └→ Aligner (Hungarian matching, bbox IoU) → VLM obj ↔ SAM3 track mapping
       │         └→ Aligned training samples (.pt in data/aligned/samples/)
       └→ enrich_timestamps.py → タイムスタンプ付き annotation JSON
            (video_metadata, coverage, clip_metadata)

Training: EventGraphDataset loads .pt → EventDecoder learns event prediction
Inference: SAM3 → FeatureExtractor → EventDecoder → postprocess → EventGraph JSON
```

#### V-JEPAパイプライン（新規）
```
Video → FrameSampler (1fps) → 16-frame clips
  ├→ V-JEPA 2.1 [frozen] → tokens → data/vjepa_features_v21_vitl/ (or vitg)
  │    backends: "hub" (PyTorch Hub, 2.1, 384px) / "hf" (HuggingFace, 2.0, 256px)
  │    variants: ViT-B (768d) / ViT-L (1024d) / ViT-g (1408d) / ViT-G (1664d), all 4608 tokens
  └→ VLMAnnotator → VLMAnnotation (objects, events)
       └→ 4b_build_vjepa_dataset.py → data/vjepa_aligned_v21_vitl/samples/*.pt

Training:
  VJEPAEventDataset → VJEPAPipeline:
    ObjectPoolingModule (Slot Attention, K=24 slots)
    → ObjectRepresentation (identity, trajectory, existence, categories)
    → VJEPAEventDecoder (cross-attn, M=20 queries, 7 prediction heads)
    → EventPredictions

  VJEPAEventGraphLoss:
    1. Slot-Object Matching (category-based Hungarian)
    2. Event Loss (remapped gt_events → EventGraphLoss)
    3. Category CE + Existence BCE
```

### Source Modules

```
src/event_graph_generation/
├── schemas/           # ObjectNode, EventEdge, EventGraph dataclasses; VLMAnnotation (Pydantic)
├── annotation/        # VLM-based synthetic annotation: vlm_annotator, prompts, alignment, validator
├── tracking/          # SAM3 wrapper (sam3_tracker) + temporal/pairwise feature extraction
├── data/              # EventGraphDataset, VJEPAEventDataset, collators, FrameSampler
├── models/            # EventDecoder, VJEPAEventDecoder, ObjectPoolingModule, VJEPAPipeline, losses
├── training/          # Trainer (AMP, early stopping, WandB), optimizer/scheduler factories
├── evaluation/        # Evaluator, metric registry (event_detection_map, action_accuracy, etc.)
├── inference/         # InferencePipeline (sliding window + NMS dedup), postprocessing
├── config.py          # Dataclass-based YAML config: Config.from_yaml(), merge(), to_yaml()
└── utils/             # seed_everything, setup_logging, save/load_checkpoint, motion, timestamps
```

### Event Decoder Model (SAM3版)

Temporal Encoder → Context Encoder (self-attention) → Event Decoder (cross-attention, learnable event queries M個) → 7つの予測ヘッド (interaction, action, agent/target/source/dest pointers, frame)。学習時はHungarian Matchingで予測↔GTの最適割当。詳細は `models/event_decoder.py` のdocstringを参照。

### V-JEPA Pipeline Model

V-JEPA tokens (hub: 2.1 384px / hf: 2.0 256px) → **ObjectPoolingModule** (InputProjection → SpatiotemporalSlotAttention K=24 slots × 3 iterations → SlotRefinement 2層self-attn → TemporalTrajectoryExtractor) → ObjectRepresentation → **VJEPAEventDecoder** (Context Encoder → Event Decoder cross-attn M=20 queries → 7予測ヘッド)。Slot Attentionはslot軸softmaxで競合的バインディング、GRU+残差MLPで更新。学習時はSlot-Object Matching（カテゴリベース）でGTインデックスをリマップしてからEventGraphLossを適用。

## Design Patterns

- **Factory**: `build_model()`, `build_optimizer()`, `build_scheduler()`, `get_metric()`
- **Dataclass Config**: 全設定がdataclass。`Config.from_yaml()` → `merge()` で実験オーバーライド
- **Metric Registry**: 文字列名でメトリクス検索 (`METRIC_REGISTRY`)
- **Set Prediction**: 固定M個のevent queries、Hungarian Matching、非自己回帰
- **Sliding Window**: 推論時にclip_length/clip_strideでオーバーラップ処理 + NMS重複排除

## Design Decisions

コードから明らかでない重要な判断:

- **オプショナル依存**: `try/except ImportError` + `_SAM3_AVAILABLE` / `_WANDB_AVAILABLE` / `_VLLM_AVAILABLE` / `_OPENAI_AVAILABLE` フラグパターン。未インストールでもgraceful degradation
- **Config追加**: dataclass にフィールド追加 → `_from_dict()` でネストdataclass対応。`configs/default.yaml` がベース設定
- **`build_model()` ファクトリ** (`models/base.py`): `config.name` でディスパッチ (`"event_decoder"`, `"vjepa_pipeline"`)、遅延インポート。新モデル追加時ここに登録
- **ポインタヘッドの K+1 規約**: `source_ptr`/`dest_ptr` は `(M, K+1)` で最後のスロットが "none"（アクション依存で任意）
- **VJEPAEventDecoder のマスキング**: ポインタlogitsのマスキングは `object_mask` 引数で外部制御。学習時は `VJEPAEventGraphLoss` がmatched slotsを含む正しいマスクを構築して `EventGraphLoss` に渡す。推論時は `existence > 0.5` のマスクを明示的に渡す
- **Slot-Object Matching**: 学習時、slotのカテゴリ予測とVLMオブジェクトのカテゴリでHungarian Matching → gt_eventsのobject indexをslot indexにリマップしてイベント損失を計算
- **`__init__.py` エクスポート**: `schemas/`, `tracking/`, `annotation/`, `inference/` は明示的 `__all__`、他は最小限

## Configuration

- `configs/default.yaml` — ベースデフォルト設定
- `configs/training.yaml` — Event Decoder学習ハイパーパラメータ
- `configs/inference.yaml` — 推論パイプライン設定（SAM3, clip, NMS）
- `configs/vlm.yaml` — Qwen 3.5 VLMモデル設定（transformersバックエンド）
- `configs/vlm_vllm.yaml` — VLLM直接バックエンド設定（テンソル並列）
- `configs/vlm_vllm_server.yaml` — VLLM server バックエンド設定（OpenAI互換API）
- `configs/vocab.yaml` — オブジェクトカテゴリ・属性語彙定義
- `configs/sam3.yaml` — SAM3トラッキングベース設定
- `configs/sam3_kitchen.yaml`, `sam3_desk.yaml`, `sam3_room.yaml` — ドメイン別SAM3設定
- `configs/actions.yaml` — アクション語彙定義（13クラス、source/destination要否フラグ）
- `configs/vjepa.yaml` — V-JEPA 2.1 ViT-L 特徴量抽出設定（384px, hub backend）
- `configs/vjepa_vitb.yaml` — V-JEPA 2.1 ViT-B 特徴量抽出設定（384px, 80M params）
- `configs/vjepa_vitg.yaml` — V-JEPA 2.1 ViT-g 特徴量抽出設定（384px, 1B params）
- `configs/vjepa_gigantic.yaml` — V-JEPA 2.1 ViT-G (Gigantic) 特徴量抽出設定（384px, 2B params）
- `configs/vjepa_training.yaml` — V-JEPAパイプライン学習設定（ViT-L 2.1ベース）
- `configs/experiment/` — 実験ごとのオーバーライド（`--override`で深いマージ）

## Key Dependencies

- `transformers` は PyPI リリースではなく git main からインストール（`pyproject.toml` 参照）
- `timm`, `einops` は V-JEPA 2.1 PyTorch Hub backend の内部依存
- `sam3` は `try/except` でオプショナル扱い（未インストールでもテスト・他機能は動作）
- `vllm` はオプショナル（`pip install vllm>=0.11.0`、transformers git mainと競合するため`pyproject.toml`には未宣言）
- `openai` はオプショナル（vllm-server バックエンド使用時のみ必要）

## Data Format

### 学習データ（`data/aligned/samples/*.pt`）

```python
{
    "object_embeddings": Tensor (N, D_emb),    # SAM3オブジェクト埋め込み
    "object_temporal": Tensor (N, T, D_geo),   # 幾何特徴（bbox, centroid, area, velocity等）
    "pairwise": Tensor (N, N, T, D_pair),      # ペアワイズ特徴（IoU, 距離, 包含関係等）
    "gt_events": list[dict]                     # GTイベント（action, agent, target, frame等）
}
```
`EventGraphDataset`がmax_objects=Kにパディングし、`event_collate_fn`でバッチ化。

### アノテーションJSON（`data/annotations_enriched/*.json`）

```jsonc
{
  "video_id": "20260316_130406_tp00001",
  "video_path": "data/resized/room/...",
  "video_metadata": {
    "video_start_time": "2026-03-16T13:04:06",  // ファイル名から算出
    "source_fps": 20.0,
    "target_fps": 1.0,
    "total_frames": 54200,
    "duration_sec": 2710.0,
    "video_end_time": "2026-03-16T13:49:16"
  },
  "coverage": {
    "total_clips": 338,
    "annotated_clips": 305,
    "motion_filtered_clips": 33,          // objects=[], events=[] のクリップ数
    "clips_with_events": 94,
    "time_range": "2026-03-16T13:04:06 ~ 2026-03-16T13:49:16",
    "motion_filtered_ranges": [...]       // 欠損時間帯のリスト
  },
  "clips": [{
    "objects": [...],
    "events": [...],
    "clip_metadata": {
      "clip_index": 0,
      "frame_indices": [0, 20, 40, ...],  // source動画のフレームインデックス
      "start_time": "2026-03-16T13:04:06",
      "end_time": "2026-03-16T13:04:21",
      "start_offset_sec": 0.0,
      "end_offset_sec": 15.0,
      "status": "annotated"               // "annotated" | "motion_filtered"
    }
  }]
}
```

`status` 判定: `objects == [] and events == []` → `"motion_filtered"`、それ以外 → `"annotated"`。
タイムスタンプ算出: `絶対時刻 = 録画開始時刻 + frame_index / source_fps`。
