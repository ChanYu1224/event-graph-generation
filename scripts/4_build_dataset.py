"""Build aligned dataset from SAM 3 tracking outputs and VLM annotations.

Constructs training samples by aligning VLM annotations to SAM 3 tracks,
then combining object features, pairwise features, and GT events into .pt files.

Usage:
    python scripts/4_build_dataset.py \\
        --sam3-dir data/sam3_outputs \\
        --annotations-dir data/annotations \\
        --output-dir data/aligned
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from itertools import combinations
from pathlib import Path

import torch
import yaml

from event_graph_generation.annotation.alignment import Aligner
from event_graph_generation.schemas.vlm_output import VLMAnnotation
from event_graph_generation.tracking.feature_extractor import FeatureExtractor
from event_graph_generation.tracking.sam3_tracker import FrameTrackingResult

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned event graph dataset.")
    parser.add_argument(
        "--sam3-dir",
        type=str,
        default="data/sam3_outputs",
        help="Directory with SAM 3 tracking .pt files.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/annotations",
        help="Directory with VLM annotation .json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/aligned",
        help="Output directory for aligned dataset.",
    )
    parser.add_argument(
        "--actions-config",
        type=str,
        default="configs/actions.yaml",
        help="YAML file mapping action names to class indices.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--temporal-window", type=int, default=16)
    return parser.parse_args()


def load_actions_config(path: str) -> dict[str, int]:
    """Load action name -> class index mapping from YAML.

    actions.yaml format is a list of dicts:
        actions:
          - name: "take_out"
            description: "..."
          - name: "put_in"
            ...
    Returns: {"take_out": 0, "put_in": 1, ...}
    """
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        actions = cfg.get("actions", [])
        if isinstance(actions, list):
            return {entry["name"]: idx for idx, entry in enumerate(actions) if "name" in entry}
        # Fallback for flat dict format
        return {str(k): int(v) for k, v in actions.items()}
    logger.warning("Actions config not found at %s, using empty mapping.", path)
    return {}


def load_sam3_results(pt_path: Path) -> list[FrameTrackingResult]:
    """Load SAM 3 tracking results from a .pt file.

    Note: weights_only=False is required here because SAM3 outputs contain
    custom dataclass objects (FrameTrackingResult, TrackedObject).
    Only load files from trusted sources (our own run_sam3_tracking.py output).
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    # If stored as a dict with 'tracking_results' key
    if isinstance(data, dict) and "tracking_results" in data:
        return data["tracking_results"]
    raise ValueError(f"Unexpected SAM 3 data format in {pt_path}")


def load_vlm_annotation(json_path: Path) -> VLMAnnotation:
    """Load VLM annotation from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return VLMAnnotation.model_validate(data)


def build_sample(
    video_id: str,
    tracking_results: list[FrameTrackingResult],
    vlm_annotation: VLMAnnotation,
    aligner: Aligner,
    feature_extractor: FeatureExtractor,
    action_map: dict[str, int],
) -> dict | None:
    """Build a single training sample from tracking + annotation.

    Returns:
        Dict with object_embeddings, object_temporal, pairwise, gt_events,
        or None if alignment fails completely.
    """
    # Align VLM objects to SAM 3 tracks
    alignment = aligner.align(tracking_results, vlm_annotation)

    if not alignment.mapping:
        logger.warning("No alignment found for video %s, skipping.", video_id)
        return None

    # Extract features
    obj_features, pair_features = feature_extractor.extract(tracking_results)

    # Get aligned track IDs in order
    track_ids = sorted(obj_features.keys())
    tid_to_idx = {tid: idx for idx, tid in enumerate(track_ids)}
    N = len(track_ids)
    T = feature_extractor.temporal_window

    if N == 0:
        return None

    # Build object_embeddings: (N, D_emb)
    D_emb = obj_features[track_ids[0]].embedding.shape[0]
    object_embeddings = torch.stack([obj_features[tid].embedding for tid in track_ids])

    # Build object_temporal: (N, T, D_geo)
    # D_geo = 4(bbox) + 2(centroid) + 1(area) + 1(presence) + 2(delta_centroid) + 1(delta_area) + 1(velocity) = 12
    object_temporal = torch.zeros(N, T, 12)
    for i, tid in enumerate(track_ids):
        of = obj_features[tid]
        # bbox_seq: (T, 4), centroid_seq: (T, 2), area_seq: (T, 1), presence_seq: (T, 1)
        object_temporal[i, :, :4] = of.bbox_seq
        object_temporal[i, :, 4:6] = of.centroid_seq
        object_temporal[i, :, 6:7] = of.area_seq
        object_temporal[i, :, 7:8] = of.presence_seq
        # deltas: (T-1, ...) -> pad to T by prepending zeros
        if T > 1:
            object_temporal[i, 1:, 8:10] = of.delta_centroid_seq
            object_temporal[i, 1:, 10:11] = of.delta_area_seq
            object_temporal[i, 1:, 11:12] = of.velocity_seq

    # Build pairwise: (N, N, T, D_pair)
    # D_pair = 1(iou) + 1(distance) + 1(containment_ij) + 1(containment_ji) + 2(rel_pos) + 1(symmetric) = 7
    pairwise = torch.zeros(N, N, T, 7)
    pair_lookup = {
        (pf.track_id_i, pf.track_id_j): pf for pf in pair_features
    }
    for i, tid_i in enumerate(track_ids):
        for j, tid_j in enumerate(track_ids):
            if i == j:
                continue
            key = (min(tid_i, tid_j), max(tid_i, tid_j))
            if key in pair_lookup:
                pf = pair_lookup[key]
                pairwise[i, j, :, 0:1] = pf.iou_seq
                pairwise[i, j, :, 1:2] = pf.distance_seq
                if tid_i == key[0]:
                    pairwise[i, j, :, 2:3] = pf.containment_ij_seq
                    pairwise[i, j, :, 3:4] = pf.containment_ji_seq
                    pairwise[i, j, :, 4:6] = pf.relative_position_seq
                else:
                    pairwise[i, j, :, 2:3] = pf.containment_ji_seq
                    pairwise[i, j, :, 3:4] = pf.containment_ij_seq
                    pairwise[i, j, :, 4:6] = -pf.relative_position_seq
                # Symmetric IoU as last feature
                pairwise[i, j, :, 6:7] = pf.iou_seq

    # Build GT events using alignment mapping (vlm_obj_id -> sam3_track_id -> idx)
    gt_events = []
    vlm_to_track = alignment.mapping  # vlm_obj_id -> sam3_track_id

    for evt in vlm_annotation.events:
        agent_tid = vlm_to_track.get(evt.agent)
        target_tid = vlm_to_track.get(evt.target)

        if agent_tid is None or target_tid is None:
            continue
        if agent_tid not in tid_to_idx or target_tid not in tid_to_idx:
            continue

        action_class = action_map.get(evt.action)
        if action_class is None:
            logger.warning("Unknown action '%s', skipping event.", evt.action)
            continue

        source_tid = vlm_to_track.get(evt.source) if evt.source else None
        dest_tid = vlm_to_track.get(evt.destination) if evt.destination else None

        gt_event = {
            "agent_track_id": tid_to_idx[agent_tid],
            "action_class": action_class,
            "target_track_id": tid_to_idx[target_tid],
            "source_track_id": tid_to_idx[source_tid] if source_tid is not None and source_tid in tid_to_idx else None,
            "dest_track_id": tid_to_idx[dest_tid] if dest_tid is not None and dest_tid in tid_to_idx else None,
            "event_frame_index": min(evt.frame, T - 1),
        }
        gt_events.append(gt_event)

    return {
        "video_id": video_id,
        "object_embeddings": object_embeddings,
        "object_temporal": object_temporal,
        "pairwise": pairwise,
        "gt_events": gt_events,
        "track_ids": track_ids,
        "alignment": {
            "mapping": alignment.mapping,
            "confidence": alignment.confidence,
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    sam3_dir = Path(args.sam3_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    splits_dir = output_dir / "splits"

    samples_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    action_map = load_actions_config(args.actions_config)
    aligner = Aligner(iou_threshold=args.iou_threshold)
    feature_extractor = FeatureExtractor(temporal_window=args.temporal_window)

    # Discover videos by matching SAM 3 .pt files with annotation .json files
    sam3_files = sorted(sam3_dir.glob("*.pt"))
    sample_ids: list[str] = []
    stats = {
        "total_videos": 0,
        "successful_alignments": 0,
        "total_events": 0,
        "total_objects": 0,
        "skipped": 0,
    }

    for sam3_file in sam3_files:
        video_id = sam3_file.stem
        annotation_file = annotations_dir / f"{video_id}.json"

        if not annotation_file.exists():
            logger.warning("No annotation for %s, skipping.", video_id)
            stats["skipped"] += 1
            continue

        stats["total_videos"] += 1

        try:
            tracking_results = load_sam3_results(sam3_file)
            vlm_annotation = load_vlm_annotation(annotation_file)
        except Exception:
            logger.exception("Failed to load data for %s", video_id)
            stats["skipped"] += 1
            continue

        sample = build_sample(
            video_id=video_id,
            tracking_results=tracking_results,
            vlm_annotation=vlm_annotation,
            aligner=aligner,
            feature_extractor=feature_extractor,
            action_map=action_map,
        )

        if sample is None:
            stats["skipped"] += 1
            continue

        # Save sample
        sample_path = samples_dir / f"{video_id}.pt"
        torch.save(sample, sample_path)
        sample_ids.append(video_id)

        stats["successful_alignments"] += 1
        stats["total_events"] += len(sample["gt_events"])
        stats["total_objects"] += sample["object_embeddings"].shape[0]

    # Create train/val/test splits
    random.seed(args.seed)
    random.shuffle(sample_ids)

    n = len(sample_ids)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val - n_test

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train : n_train + n_val]
    test_ids = sample_ids[n_train + n_val :]

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, "w") as f:
            for sid in sorted(ids):
                f.write(f"{sid}\n")
        logger.info("Split '%s': %d samples -> %s", split_name, len(ids), split_file)

    # Save metadata
    stats["train_count"] = len(train_ids)
    stats["val_count"] = len(val_ids)
    stats["test_count"] = len(test_ids)

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset built: %s", json.dumps(stats, indent=2))
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
