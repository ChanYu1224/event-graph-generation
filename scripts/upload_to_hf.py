"""Upload best model checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

VARIANTS = {
    "vitb": {
        "repo_suffix": "event-graph-vitb",
        "checkpoint_glob": "vjepa_vitb_*/best.pt",
        "hub_model_name": "vjepa2_1_vit_base_384",
        "hidden_size": 768,
        "override_config": "configs/experiment/vjepa_vitb.yaml",
        "display_name": "ViT-B",
    },
    "vitg": {
        "repo_suffix": "event-graph-vitg",
        "checkpoint_glob": "vjepa_vitg_*/best.pt",
        "hub_model_name": "vjepa2_1_vit_giant_384",
        "hidden_size": 1408,
        "override_config": "configs/experiment/vjepa_vitg.yaml",
        "display_name": "ViT-G",
    },
    "vitl": {
        "repo_suffix": "event-graph-vitl",
        "checkpoint_glob": "vjepa_vitl_*/best.pt",
        "hub_model_name": "vjepa2_1_vit_large_384",
        "hidden_size": 1024,
        "override_config": None,
        "display_name": "ViT-L",
    },
}

BASE_CONFIG = "configs/vjepa_training.yaml"

MIT_LICENSE = """MIT License

Copyright (c) 2026 Yuchn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _find_best_checkpoint(checkpoint_dir: Path, variant: dict) -> Path:
    """Find the best.pt checkpoint for a variant."""
    pattern = str(checkpoint_dir / variant["checkpoint_glob"])
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
    if len(matches) > 1:
        matches.sort()
        logger.warning("Multiple matches found, using latest: %s", matches[-1])
    return Path(matches[-1])


def _strip_optimizer(checkpoint_path: Path, output_path: Path) -> dict:
    """Load checkpoint and save model_state_dict only."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    epoch = ckpt.get("epoch", "unknown")
    torch.save(state_dict, output_path)
    return {"epoch": epoch, "num_params": sum(p.numel() for p in state_dict.values())}


def _merge_config(base_path: str, override_path: str | None) -> dict:
    """Load and merge YAML configs."""
    with open(base_path) as f:
        config = yaml.safe_load(f)
    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            override = yaml.safe_load(f)
        for key, value in override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    return config


def _generate_model_card(
    variant_key: str,
    variant: dict,
    config: dict,
    meta: dict,
    namespace: str,
) -> str:
    """Generate a HuggingFace model card README.md."""
    display = variant["display_name"]
    repo_name = f"{namespace}/{variant['repo_suffix']}"
    num_params_m = meta["num_params"] / 1e6

    model_config = config.get("model", {})
    vjepa_config = config.get("vjepa", {})
    training_config = config.get("training", {})
    pooling_config = model_config.get("object_pooling", {})

    num_slots = pooling_config.get("num_slots", 24)
    num_queries = model_config.get("num_event_queries", 20)
    num_actions = model_config.get("num_actions", 13)
    d_model = model_config.get("d_model", 256)
    hub_model = vjepa_config.get("hub_model_name", "N/A")
    lr = training_config.get("learning_rate", "1e-4")
    weight_decay = training_config.get("weight_decay", "1e-4")
    optimizer = training_config.get("optimizer", "adamw")
    scheduler = training_config.get("scheduler", "cosine_warmup")
    scheduler_params = training_config.get("scheduler_params", {})
    warmup_epochs = scheduler_params.get("warmup_epochs", 5)
    patience = training_config.get("early_stopping_patience", 15)
    loss_weights = training_config.get("loss_weights", {})

    return f"""---
license: mit
tags:
  - video-understanding
  - event-graph
  - vjepa
  - slot-attention
  - pytorch
  - set-prediction
  - hungarian-matching
library_name: pytorch
pipeline_tag: video-classification
---

# Event Graph Generation — {display}

## Model Overview

動画から構造化されたイベントグラフを予測するモデルです。動画中の「**誰が**・**何を**・**どこから**・**どこへ**」を構造化 JSON として出力します。

テキスト生成ではなく、DETR 風の**セット予測**（Hungarian Matching）により、イベントグラフを直接出力する設計です。

## Base Model

[V-JEPA 2.1](https://github.com/facebookresearch/vjepa) {display} (`{hub_model}`) を映像特徴抽出のバックボーンとして使用しています。V-JEPA は Meta が開発した自己教師あり映像表現学習モデルで、時空間トークンを出力します。

本モデル（Event Decoder 部分）は V-JEPA の出力トークンを入力とし、Object Pooling と Event Decoder の 2 段階でイベントグラフを予測します。V-JEPA 自体の重みは本チェックポイントに**含まれません**（別途 PyTorch Hub からロードされます）。

## Model Details

| 項目 | 値 |
|---|---|
| パラメータ数 (Event Decoder) | **{num_params_m:.1f}M** |
| V-JEPA backbone | `{hub_model}` |
| V-JEPA hidden_size | {variant["hidden_size"]} |
| Object Pooling | Slot Attention (K={num_slots} slots, {pooling_config.get("num_iterations", 3)} iterations) |
| Event Decoder | DETR 風 cross-attention (M={num_queries} event queries) |
| d_model | {d_model} |
| Action classes | {num_actions} |
| Best epoch | {meta["epoch"]} |

### Architecture

```
Video → V-JEPA 2.1 {display} → spatiotemporal tokens (B, S, {variant["hidden_size"]})
  → ObjectPoolingModule (Slot Attention, K={num_slots} slots)
    → ObjectRepresentation (identity, trajectory, existence, categories)
      → VJEPAEventDecoder (M={num_queries} event queries, cross-attention)
        → 7 Prediction Heads → EventGraph JSON
```

### Prediction Heads

| Head | Shape | Description |
|---|---|---|
| interaction | (M, 1) | イベントが有効か (BCE) |
| action | (M, {num_actions}) | アクション分類 (13 クラス) |
| agent_ptr | (M, K) | 行為者スロットへのポインタ |
| target_ptr | (M, K) | 対象スロットへのポインタ |
| source_ptr | (M, K+1) | 取り出し元へのポインタ (最後 = "none") |
| dest_ptr | (M, K+1) | 格納先へのポインタ (最後 = "none") |
| frame | (M, T) | イベント発生フレーム |

## Training

### Fine-tuning

- **ファインチューニング**: あり（V-JEPA backbone は frozen、Event Decoder 部分のみ学習）
- **学習手法**: Full fine-tuning（Event Decoder 全体を学習。LoRA 等は未使用）
- **Optimizer**: {optimizer} (lr={lr}, weight_decay={weight_decay})
- **Scheduler**: {scheduler} (warmup {warmup_epochs} epochs)
- **Early stopping**: patience={patience}
- **AMP**: bfloat16 mixed precision
- **損失関数**: Hungarian Matching によるセット予測損失

**Loss weights:**

| Component | Weight |
|---|---|
| interaction | {loss_weights.get("interaction", 2.0)} |
| action | {loss_weights.get("action", 1.0)} |
| agent_ptr | {loss_weights.get("agent_ptr", 1.0)} |
| target_ptr | {loss_weights.get("target_ptr", 1.0)} |
| source_ptr | {loss_weights.get("source_ptr", 0.5)} |
| dest_ptr | {loss_weights.get("dest_ptr", 0.5)} |
| frame | {loss_weights.get("frame", 0.5)} |

### Training Data

- **データ**: 室内環境の録画動画（デスク・キッチン・部屋）
- **アノテーション**: Qwen 3.5 VLM による合成アノテーション（人手ラベルなし）
- **フレームレート**: 1 FPS でサンプリング、16 フレーム/クリップ、50% オーバーラップ
- **オブジェクトカテゴリ**: 28 カテゴリ + unknown（person, hand, chair, desk, laptop, monitor 等）
- **アクション語彙**: 13 クラス（take_out, put_in, place_on, pick_up, hand_over, open, close, use, move, attach, detach, inspect, no_event）

## Intended Use

### Intended Use Cases

- 製造・組立作業の動画からの作業イベント自動抽出
- 室内行動の構造的記録（誰が何をどこに置いたか等）
- 動画理解研究のための構造化アノテーション自動生成
- IoT / スマートホーム環境での行動ログ生成

### Out-of-Scope Use

- 個人の監視・追跡を目的とした利用
- 屋外・交通・医療など、学習データに含まれないドメインでの高精度な利用
- リアルタイム処理が必要なシステム（V-JEPA backbone の推論コストが高い）
- セキュリティ判断や法的判断の根拠としての利用

## Evaluation

本モデルは以下のメトリクスで評価されています:

| Metric | Description |
|---|---|
| event_detection_mAP | イベント検出の平均適合率 |
| action_accuracy | アクション分類精度 |
| pointer_accuracy | agent/target ポインタの正解率 |
| frame_mae | イベントフレーム予測の平均絶対誤差 |
| graph_f1 | EventGraph 全体の F1 スコア |

> **注意**: 本モデルは VLM 合成アノテーションで学習されており、人手アノテーションによるベンチマークスコアは未計測です。

## Inference

### End-to-End Inference (推奨)

```bash
# リポジトリをクローン
git clone https://github.com/ChanYu1224/event-graph-generation.git
cd event-graph-generation
uv sync

# 推論実行
uv run python scripts/6_run_inference.py \\
  --video your_video.mp4 \\
  --checkpoint path/to/model.pt \\
  --config configs/vjepa_training.yaml \\
  --vjepa-config configs/vjepa.yaml \\
  --output output/event_graph.json
```

### Python API

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="{repo_name}", filename="model.pt")
config_path = hf_hub_download(repo_id="{repo_name}", filename="config.yaml")

# Build model
from event_graph_generation.config import Config
from event_graph_generation.models.base import build_model

config = Config.from_yaml(config_path)
model = build_model(config.model, vjepa_config=config.vjepa)

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Forward pass (vjepa_tokens: pre-extracted V-JEPA features)
# vjepa_tokens shape: (batch_size, num_tokens, hidden_size)
with torch.no_grad():
    obj_repr, predictions = model(vjepa_tokens)
```

## Limitations

### Bias

- 学習データは室内環境（デスク・キッチン・部屋）に限定されており、屋外や工場環境での精度は未検証
- VLM 合成アノテーションに依存しているため、Qwen 3.5 のバイアスを継承する可能性がある
- オブジェクトカテゴリは 28 種類に限定されており、未知カテゴリのオブジェクトは "unknown" として扱われる

### Limitations

- V-JEPA backbone は frozen のため、ドメイン固有の映像表現への適応は限定的
- 13 種類のアクション語彙に限定されており、語彙外の行動は検出できない
- 1 FPS サンプリングのため、1 秒未満の高速なイベントは見逃す可能性がある
- 推論には CUDA 対応 GPU が必要（V-JEPA backbone + Event Decoder）
- 長時間動画ではスライディングウィンドウ処理のため、メモリ使用量と処理時間が線形に増加

## License

MIT License

## Citation

```bibtex
@software{{event_graph_generation_2026,
  title = {{Event Graph Generation: Structured Event Prediction from Video}},
  author = {{Yuchn}},
  year = {{2026}},
  url = {{https://github.com/ChanYu1224/event-graph-generation}},
  license = {{MIT}}
}}
```

## Links

- **Repository**: [ChanYu1224/event-graph-generation](https://github.com/ChanYu1224/event-graph-generation)
- **V-JEPA**: [facebookresearch/vjepa](https://github.com/facebookresearch/vjepa)
"""


def upload_variant(
    variant_key: str,
    namespace: str,
    checkpoint_dir: Path,
    dry_run: bool = False,
) -> str:
    """Upload a single model variant to HuggingFace Hub."""
    variant = VARIANTS[variant_key]
    repo_id = f"{namespace}/{variant['repo_suffix']}"
    logger.info("Processing %s → %s", variant_key, repo_id)

    # Find checkpoint
    ckpt_path = _find_best_checkpoint(checkpoint_dir, variant)
    logger.info("Checkpoint: %s", ckpt_path)

    # Merge config
    config = _merge_config(BASE_CONFIG, variant["override_config"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Strip optimizer and save model weights
        meta = _strip_optimizer(ckpt_path, tmpdir / "model.pt")
        logger.info(
            "Model: %.1fM params, epoch %s",
            meta["num_params"] / 1e6,
            meta["epoch"],
        )

        # 2. Save merged config
        with open(tmpdir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # 3. Generate model card
        model_card = _generate_model_card(
            variant_key, variant, config, meta, namespace
        )
        (tmpdir / "README.md").write_text(model_card)

        # 4. Write LICENSE
        (tmpdir / "LICENSE").write_text(MIT_LICENSE)

        if dry_run:
            logger.info("Dry run — files prepared in %s:", tmpdir)
            for f in sorted(tmpdir.iterdir()):
                size = f.stat().st_size
                logger.info("  %s (%s)", f.name, _human_size(size))
            # Copy to persistent location for inspection
            inspect_dir = Path(f"_hf_upload_preview/{variant_key}")
            if inspect_dir.exists():
                shutil.rmtree(inspect_dir)
            shutil.copytree(tmpdir, inspect_dir)
            logger.info("Preview saved to %s", inspect_dir)
            return f"(dry-run) {repo_id}"

        # 5. Upload to HF Hub
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {variant['display_name']} event graph model",
        )
        logger.info("Uploaded to https://huggingface.co/%s", repo_id)
        return f"https://huggingface.co/{repo_id}"


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload best model checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["vitb", "vitg", "vitl", "all"],
        help="Model variant to upload (default: all)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="Yuchn",
        help="HuggingFace namespace (user or org)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint subdirectories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files without uploading",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    results = []
    for v in variants:
        url = upload_variant(v, args.namespace, checkpoint_dir, dry_run=args.dry_run)
        results.append(url)

    print("\n=== Upload Results ===")
    for url in results:
        print(f"  {url}")


if __name__ == "__main__":
    main()
