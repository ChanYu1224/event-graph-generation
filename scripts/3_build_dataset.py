"""Build V-JEPA training dataset from VLM annotations + V-JEPA features.

Links VLM annotation clips with pre-extracted V-JEPA feature .pt files,
constructs gt_events with object category indices, and creates train/val/test splits.

Usage:
    uv run python scripts/4b_build_vjepa_dataset.py \
        --annotations-dir data/annotations_enriched \
        --features-dir data/vjepa_features \
        --output-dir data/vjepa_aligned \
        --vocab configs/vocab.yaml \
        --actions configs/actions.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_category_vocab(vocab_path: str) -> dict[str, int]:
    """Load category vocabulary from YAML and return name -> index mapping.

    Args:
        vocab_path: Path to vocab.yaml.

    Returns:
        Dict mapping category name to index (last index = unknown).
    """
    with open(vocab_path) as f:
        vocab = yaml.safe_load(f)
    categories = vocab["categories"]
    mapping = {name: idx for idx, name in enumerate(categories)}
    # Add unknown as last index
    mapping["unknown"] = len(categories)
    return mapping


def load_action_vocab(actions_path: str) -> dict[str, int]:
    """Load action vocabulary from YAML.

    Args:
        actions_path: Path to actions.yaml.

    Returns:
        Dict mapping action name to index.
    """
    with open(actions_path) as f:
        data = yaml.safe_load(f)
    return {a["name"]: idx for idx, a in enumerate(data["actions"])}


def extract_category_from_obj_id(obj_id: str) -> str:
    """Extract category name from object ID (e.g. 'person_01' -> 'person').

    Args:
        obj_id: Object identifier string.

    Returns:
        Category name.
    """
    parts = obj_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return obj_id


def process_clip(
    clip: dict,
    vjepa_path: Path,
    category_vocab: dict[str, int],
    action_vocab: dict[str, int],
    video_id: str,
) -> dict | None:
    """Process a single clip into a training sample.

    Args:
        clip: Clip dict from annotation JSON.
        vjepa_path: Path to the V-JEPA .pt file for this clip.
        category_vocab: Category name -> index mapping.
        action_vocab: Action name -> index mapping.
        video_id: Video identifier.

    Returns:
        Sample dict or None if clip should be skipped.
    """
    if not vjepa_path.exists():
        return None

    # Load V-JEPA features
    vjepa_data = torch.load(vjepa_path, map_location="cpu", weights_only=True)
    vjepa_tokens = vjepa_data["vjepa_tokens"]

    objects = clip.get("objects", [])
    events = clip.get("events", [])

    # Skip motion-filtered clips (no objects, no events)
    clip_meta = clip.get("clip_metadata", {})
    if clip_meta.get("status") == "motion_filtered":
        return None

    # Build object index mapping: obj_id -> local_index
    obj_id_to_idx: dict[str, int] = {}
    gt_object_categories: list[int] = []
    for i, obj in enumerate(objects):
        obj_id = obj.get("id", obj.get("obj_id", f"obj_{i}"))
        obj_id_to_idx[obj_id] = i
        cat_name = extract_category_from_obj_id(obj_id)
        cat_idx = category_vocab.get(cat_name, category_vocab["unknown"])
        gt_object_categories.append(cat_idx)

    # Build gt_events
    gt_events: list[dict] = []
    for evt in events:
        action_name = evt.get("action", evt.get("action_type", ""))
        if action_name not in action_vocab:
            logger.debug("Unknown action '%s', skipping event", action_name)
            continue

        agent_id = evt.get("agent", evt.get("agent_id", ""))
        target_id = evt.get("target", evt.get("target_id", ""))

        if agent_id not in obj_id_to_idx or target_id not in obj_id_to_idx:
            continue

        source_id = evt.get("source", evt.get("source_id"))
        dest_id = evt.get("destination", evt.get("dest_id"))

        # Clamp frame index to valid range [0, clip_length - 1]
        frame_idx = evt.get("frame_index", evt.get("frame", 0))
        frame_idx = max(0, min(frame_idx, 15))  # clip_length=16 → max index 15

        gt_event = {
            "action_class": action_vocab[action_name],
            "agent_track_id": obj_id_to_idx[agent_id],
            "target_track_id": obj_id_to_idx[target_id],
            "source_track_id": obj_id_to_idx.get(source_id) if source_id else None,
            "dest_track_id": obj_id_to_idx.get(dest_id) if dest_id else None,
            "event_frame_index": frame_idx,
        }
        gt_events.append(gt_event)

    clip_index = clip_meta.get("clip_index", 0)

    return {
        "vjepa_tokens": vjepa_tokens,
        "gt_events": gt_events,
        "gt_object_categories": gt_object_categories,
        "num_objects": len(objects),
        "video_id": video_id,
        "clip_index": clip_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V-JEPA training dataset")
    parser.add_argument("--annotations-dir", type=str, default="data/annotations_enriched")
    parser.add_argument("--features-dir", type=str, default="data/vjepa_features")
    parser.add_argument("--output-dir", type=str, default="data/vjepa_aligned")
    parser.add_argument("--vocab", type=str, default="configs/vocab.yaml")
    parser.add_argument("--actions", type=str, default="configs/actions.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()

    annotations_dir = Path(args.annotations_dir)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    splits_dir = output_dir / "splits"
    samples_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    category_vocab = load_category_vocab(args.vocab)
    action_vocab = load_action_vocab(args.actions)
    logger.info("Loaded %d categories, %d actions", len(category_vocab), len(action_vocab))

    # Process all annotation files
    annotation_files = sorted(annotations_dir.glob("*.json"))
    logger.info("Found %d annotation files", len(annotation_files))

    sample_ids: list[str] = []
    total_events = 0

    for ann_path in annotation_files:
        with open(ann_path) as f:
            ann = json.load(f)

        video_id = ann.get("video_id", ann_path.stem)
        clips = ann.get("clips", [])
        video_features_dir = features_dir / video_id

        for clip in clips:
            clip_meta = clip.get("clip_metadata", {})
            clip_idx = clip_meta.get("clip_index", 0)
            vjepa_path = video_features_dir / f"clip_{clip_idx:04d}.pt"

            sample = process_clip(clip, vjepa_path, category_vocab, action_vocab, video_id)
            if sample is None:
                continue

            sample_id = f"{video_id}_clip_{clip_idx:04d}"
            sample_path = samples_dir / f"{sample_id}.pt"
            torch.save(sample, sample_path)
            sample_ids.append(sample_id)
            total_events += len(sample["gt_events"])

    logger.info(
        "Created %d samples with %d total events", len(sample_ids), total_events
    )

    # Create splits
    random.seed(args.seed)
    random.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train : n_train + n_val]
    test_ids = sample_ids[n_train + n_val :]

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, "w") as f:
            for sid in sorted(ids):
                f.write(f"{sid}\n")
        logger.info("Split %s: %d samples", split_name, len(ids))


if __name__ == "__main__":
    main()
