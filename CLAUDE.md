# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Event Decoder Model

Temporal Encoder → Context Encoder (self-attention) → Event Decoder (cross-attention, learnable event queries M個) → 7つの予測ヘッド (interaction, action, agent/target/source/dest pointers, frame)。学習時はHungarian Matchingで予測↔GTの最適割当。詳細は `models/event_decoder.py` のdocstringを参照。

## Design Patterns

- **Factory**: `build_model()`, `build_optimizer()`, `build_scheduler()`, `get_metric()`
- **Dataclass Config**: 全設定がdataclass。`Config.from_yaml()` → `merge()` で実験オーバーライド
- **Metric Registry**: 文字列名でメトリクス検索 (`METRIC_REGISTRY`)
- **Set Prediction**: 固定M個のevent queries、Hungarian Matching、非自己回帰
- **Sliding Window**: 推論時にclip_length/clip_strideでオーバーラップ処理 + NMS重複排除

## Design Decisions

コードから明らかでない重要な判断:

- **オプショナル依存**: `try/except ImportError` + `_SAM3_AVAILABLE` / `_WANDB_AVAILABLE` フラグパターン。未インストールでもgraceful degradation
- **Config追加**: dataclass にフィールド追加 → `_from_dict()` でネストdataclass対応。`configs/default.yaml` がベース設定
- **`build_model()` ファクトリ** (`models/base.py`): `config.name` でディスパッチ、遅延インポート。新モデル追加時ここに登録
- **ポインタヘッドの K+1 規約**: `source_ptr`/`dest_ptr` は `(M, K+1)` で最後のスロットが "none"（アクション依存で任意）
- **`__init__.py` エクスポート**: `schemas/`, `tracking/`, `annotation/`, `inference/` は明示的 `__all__`、他は最小限

## Configuration

- `configs/default.yaml` — ベースデフォルト設定
- `configs/training.yaml` — Event Decoder学習ハイパーパラメータ
- `configs/inference.yaml` — 推論パイプライン設定（SAM3, clip, NMS）
- `configs/vlm.yaml` — Qwen 3.5 VLMモデル設定
- `configs/sam3.yaml` — SAM3トラッキングベース設定
- `configs/sam3_kitchen.yaml`, `sam3_desk.yaml`, `sam3_room.yaml` — ドメイン別SAM3設定
- `configs/actions.yaml` — アクション語彙定義（13クラス、source/destination要否フラグ）
- `configs/experiment/` — 実験ごとのオーバーライド（`--override`で深いマージ）

## Key Dependencies

- `transformers` は PyPI リリースではなく git main からインストール（`pyproject.toml` 参照）
- `sam3` は `try/except` でオプショナル扱い（未インストールでもテスト・他機能は動作）

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
