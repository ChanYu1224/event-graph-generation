# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

画像から構造化されたイベント情報を抽出・生成する深層学習フレームワーク。Transformer系のVisionモデル（Qwen VL等）を活用し、画像からイベントグラフをJSON形式で生成する。

## Commands

```bash
# 依存関係インストール
uv sync

# テスト実行
pytest
pytest tests/test_config.py          # 特定ファイル
pytest -k test_accuracy              # パターン指定

# 学習
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --override configs/experiment/my_exp.yaml
python scripts/train.py --config configs/default.yaml --resume checkpoints/epoch_0050.pt

# 評価
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/epoch_0050.pt

# 推論
python scripts/predict.py --config configs/default.yaml --checkpoint checkpoints/epoch_0050.pt --input input.pt

# 画像からイベント抽出
python scripts/vision_understanding.py --input-dir data/images/ --output-dir output/ --model-name qwen2-vl-7b-instruct

# VRAMベンチマーク
python scripts/benchmark_vram.py
```

## Architecture

```
src/event_graph_generation/
├── config.py          # Dataclass-based YAML config (Config.from_yaml(), merge(), to_yaml())
├── data/
│   ├── dataset.py     # BaseDataset (abstract: _load_samples, __len__, __getitem__)
│   ├── collator.py    # Batch dataclass + collate_fn
│   └── transforms.py  # build_transforms() factory
├── models/
│   └── base.py        # BaseModel + build_model() factory
├── training/
│   ├── trainer.py     # Training loop with W&B, checkpointing, grad clipping
│   └── optimizer.py   # build_optimizer/build_scheduler (adam|adamw|sgd, cosine|step|none)
├── evaluation/
│   ├── evaluator.py   # Evaluator class
│   └── metrics.py     # METRIC_REGISTRY + get_metric() factory
└── utils/
    ├── seed.py        # seed_everything()
    ├── logging.py     # setup_logging()
    └── io.py          # save_checkpoint / load_checkpoint
```

**設計パターン**: Factory（build_model, build_optimizer, get_metric）、Dataclass Config、Metric Registry

## Key Dependencies

- **uv** でパッケージ管理、Python >= 3.13
- transformers (git main branch), accelerate, torch >= 2.10
- qwen-vl-utils（Qwen VLモデル用）
- wandb（実験追跡）
- bitsandbytes（量子化）

## Configuration

- `configs/default.yaml` がベース設定
- `configs/experiment/` に実験ごとのオーバーライドを配置
- `--override` フラグで深いマージが可能
