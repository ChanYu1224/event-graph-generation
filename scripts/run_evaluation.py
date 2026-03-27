"""End-to-end evaluation: build aligned dataset on-the-fly and run metrics.

Skips the intermediate disk write by processing annotations + features
directly into batches, then running the Evaluator.

Usage:
    uv run python scripts/run_evaluation.py \
        --config configs/vjepa_training.yaml \
        --checkpoint checkpoints/vjepa_vitl_20260323_094236/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.evaluation.evaluator import Evaluator
from event_graph_generation.models.base import build_model
from event_graph_generation.utils.io import load_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocab loading (reused from scripts/3_build_dataset.py)
# ---------------------------------------------------------------------------

def _load_category_vocab(vocab_path: str) -> dict[str, int]:
    with open(vocab_path) as f:
        vocab = yaml.safe_load(f)
    mapping = {name: idx for idx, name in enumerate(vocab["categories"])}
    mapping["unknown"] = len(vocab["categories"])
    return mapping


def _load_action_vocab(actions_path: str) -> dict[str, int]:
    with open(actions_path) as f:
        data = yaml.safe_load(f)
    return {a["name"]: idx for idx, a in enumerate(data["actions"])}


def _extract_category(obj_id: str) -> str:
    parts = obj_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return obj_id


# ---------------------------------------------------------------------------
# On-the-fly dataset
# ---------------------------------------------------------------------------

def _process_clip(
    clip: dict,
    vjepa_path: Path,
    category_vocab: dict[str, int],
    action_vocab: dict[str, int],
) -> dict | None:
    clip_meta = clip.get("clip_metadata", {})
    if clip_meta.get("status") == "motion_filtered":
        return None

    if not vjepa_path.exists():
        return None

    objects = clip.get("objects", [])
    events = clip.get("events", [])

    obj_id_to_idx: dict[str, int] = {}
    gt_object_categories: list[int] = []
    for i, obj in enumerate(objects):
        obj_id = obj.get("id", obj.get("obj_id", f"obj_{i}"))
        obj_id_to_idx[obj_id] = i
        cat_name = _extract_category(obj_id)
        cat_idx = category_vocab.get(cat_name, category_vocab["unknown"])
        gt_object_categories.append(cat_idx)

    gt_events: list[dict] = []
    for evt in events:
        action_name = evt.get("action", evt.get("action_type", ""))
        if action_name not in action_vocab:
            continue
        agent_id = evt.get("agent", evt.get("agent_id", ""))
        target_id = evt.get("target", evt.get("target_id", ""))
        if agent_id not in obj_id_to_idx or target_id not in obj_id_to_idx:
            continue

        source_id = evt.get("source", evt.get("source_id"))
        dest_id = evt.get("destination", evt.get("dest_id"))
        frame_idx = max(0, min(evt.get("frame_index", evt.get("frame", 0)), 15))

        gt_events.append({
            "action_class": action_vocab[action_name],
            "agent_track_id": obj_id_to_idx[agent_id],
            "target_track_id": obj_id_to_idx[target_id],
            "source_track_id": obj_id_to_idx.get(source_id) if source_id else None,
            "dest_track_id": obj_id_to_idx.get(dest_id) if dest_id else None,
            "event_frame_index": frame_idx,
        })

    return {
        "vjepa_path": vjepa_path,
        "gt_events": gt_events,
        "gt_object_categories": gt_object_categories,
        "num_objects": len(objects),
    }


class LazyVJEPADataset(Dataset):
    """Lazy-loading dataset: stores only metadata, loads .pt on __getitem__."""

    def __init__(
        self,
        annotations_dir: Path,
        features_dir: Path,
        category_vocab: dict[str, int],
        action_vocab: dict[str, int],
        sample_ids: list[str] | None = None,
    ) -> None:
        self.entries: list[dict] = []
        self._build(annotations_dir, features_dir, category_vocab, action_vocab,
                     sample_ids)

    def _build(
        self,
        annotations_dir: Path,
        features_dir: Path,
        category_vocab: dict[str, int],
        action_vocab: dict[str, int],
        sample_ids: list[str] | None,
    ) -> None:
        allowed: set[str] | None = set(sample_ids) if sample_ids else None
        total_events = 0

        for ann_path in sorted(annotations_dir.glob("*.json")):
            with open(ann_path) as f:
                ann = json.load(f)

            video_id = ann.get("video_id", ann_path.stem)
            video_features_dir = features_dir / video_id

            for clip in ann.get("clips", []):
                clip_meta = clip.get("clip_metadata", {})
                clip_idx = clip_meta.get("clip_index", 0)

                if allowed is not None:
                    sid = f"{video_id}_clip_{clip_idx:04d}"
                    if sid not in allowed:
                        continue

                vjepa_path = video_features_dir / f"clip_{clip_idx:04d}.pt"
                sample = _process_clip(clip, vjepa_path, category_vocab, action_vocab)
                if sample is not None:
                    self.entries.append(sample)
                    total_events += len(sample["gt_events"])

        logger.info("Built dataset: %d samples, %d events", len(self.entries), total_events)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        vjepa_data = torch.load(entry["vjepa_path"], map_location="cpu", weights_only=True)
        return {
            "vjepa_tokens": vjepa_data["vjepa_tokens"],
            "gt_events": entry["gt_events"],
            "gt_object_categories": entry["gt_object_categories"],
            "num_objects": entry["num_objects"],
        }


# ---------------------------------------------------------------------------
# Collator / Batch
# ---------------------------------------------------------------------------

@dataclass
class VJEPAEvalBatch:
    vjepa_tokens: torch.Tensor  # (B, S, D)
    gt_events: list[list[dict]]
    gt_object_categories: list[list[int]]
    num_objects: list[int]

    def to(self, device: str | torch.device) -> VJEPAEvalBatch:
        self.vjepa_tokens = self.vjepa_tokens.to(device)
        return self


def _collate_fn(samples: list[dict]) -> VJEPAEvalBatch:
    return VJEPAEvalBatch(
        vjepa_tokens=torch.stack([s["vjepa_tokens"] for s in samples]),
        gt_events=[s["gt_events"] for s in samples],
        gt_object_categories=[s["gt_object_categories"] for s in samples],
        num_objects=[s["num_objects"] for s in samples],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA pipeline")
    parser.add_argument("--config", default="configs/vjepa_training.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vjepa_vitl_20260323_094236/best.pt",
    )
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--features-dir", default="data/vjepa_features_v21_vitl")
    parser.add_argument("--vocab", default="configs/vocab.yaml")
    parser.add_argument("--actions", default="configs/actions.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data to use as test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test",
                        choices=["test", "val", "all"],
                        help="Which split to evaluate on")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Load vocabs
    category_vocab = _load_category_vocab(args.vocab)
    action_vocab = _load_action_vocab(args.actions)
    logger.info("Categories: %d, Actions: %d", len(category_vocab), len(action_vocab))

    # Discover all available sample IDs to create splits
    logger.info("Scanning annotations and features...")
    annotations_dir = Path(args.annotations_dir)
    features_dir = Path(args.features_dir)

    all_sample_ids: list[str] = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        with open(ann_path) as f:
            ann = json.load(f)
        video_id = ann.get("video_id", ann_path.stem)
        video_features_dir = features_dir / video_id
        for clip in ann.get("clips", []):
            clip_meta = clip.get("clip_metadata", {})
            if clip_meta.get("status") == "motion_filtered":
                continue
            clip_idx = clip_meta.get("clip_index", 0)
            vjepa_path = video_features_dir / f"clip_{clip_idx:04d}.pt"
            if vjepa_path.exists():
                all_sample_ids.append(f"{video_id}_clip_{clip_idx:04d}")

    logger.info("Found %d valid samples total", len(all_sample_ids))

    # Create deterministic split
    rng = random.Random(args.seed)
    shuffled = list(all_sample_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.test_ratio)
    n_train = n_total - n_test - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))

    # Select split
    if args.split == "test":
        eval_ids = test_ids
    elif args.split == "val":
        eval_ids = val_ids
    else:
        eval_ids = all_sample_ids

    logger.info("Evaluating on %s split: %d samples", args.split, len(eval_ids))

    # Build dataset
    dataset = LazyVJEPADataset(
        annotations_dir=annotations_dir,
        features_dir=features_dir,
        category_vocab=category_vocab,
        action_vocab=action_vocab,
        sample_ids=eval_ids,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    # Build and load model
    logger.info("Loading model from %s", args.checkpoint)
    model = build_model(config.model, vjepa_config=config.vjepa)
    load_checkpoint(args.checkpoint, model, device=args.device)
    model.to(args.device)
    model.eval()

    # Run evaluation
    evaluator = Evaluator(config.evaluation, device=args.device)
    logger.info("Running evaluation...")
    results = evaluator.evaluate(model, dataloader)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print(f"Model: {args.checkpoint}")
    print(f"Split: {args.split} ({len(eval_ids)} samples)")
    print("=" * 60)
    for name, value in results.items():
        print(f"  {name:<25s} {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
