"""Batch SAM 3 tracking script with resume support.

Usage:
    # From pre-extracted frames (recommended — low RAM usage):
    python scripts/2_run_sam3_tracking.py --config configs/sam3.yaml --frames-dir data/frames/room --output-dir data/sam3_outputs --resume

    # From video files directly (legacy — high RAM usage):
    python scripts/2_run_sam3_tracking.py --config configs/sam3.yaml --video-dir data/videos --output-dir data/sam3_outputs --resume
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from event_graph_generation.tracking import FeatureExtractor, SAM3Tracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 3 tracking on video files.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam3.yaml",
        help="Path to SAM 3 config YAML.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing video files (legacy mode, high RAM usage).",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory containing pre-extracted frames (from 1b_extract_frames.py). Recommended.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sam3_outputs",
        help="Directory to save tracking outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed videos.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index for multi-GPU parallel processing.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU parallel processing.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    logger.warning("Config file %s not found, using defaults.", config_path)
    return {}


def discover_videos(video_dir: str) -> list[Path]:
    """Find all video files in the given directory (recursive)."""
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        logger.error("Video directory does not exist: %s", video_dir)
        return []

    extensions = {".mp4", ".avi", ".mov"}
    videos = sorted(
        p for p in video_dir_path.rglob("*") if p.suffix.lower() in extensions
    )
    logger.info("Discovered %d video files in %s", len(videos), video_dir)
    return videos


def discover_frame_dirs(frames_dir: str) -> list[Path]:
    """Find all frame directories (containing metadata.json) in the given directory."""
    frames_dir_path = Path(frames_dir)
    if not frames_dir_path.exists():
        logger.error("Frames directory does not exist: %s", frames_dir)
        return []

    dirs = sorted(
        p.parent for p in frames_dir_path.glob("*/metadata.json")
    )
    logger.info("Discovered %d frame directories in %s", len(dirs), frames_dir)
    return dirs


def load_extracted_frames(
    frame_dir: Path,
) -> tuple[list[np.ndarray], list[int]]:
    """Load pre-extracted frames from a directory with metadata.json.

    Reads JPEG files one at a time to keep memory usage proportional to
    the number of extracted frames (not the full video).

    Args:
        frame_dir: Directory containing frame_*.jpg and metadata.json.

    Returns:
        Tuple of (frames, frame_indices).
    """
    metadata_path = frame_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    frame_indices: list[int] = metadata["frame_indices"]
    frame_filenames: list[str] | None = metadata.get("frame_filenames")
    frames: list[np.ndarray] = []

    for i, idx in enumerate(frame_indices):
        if frame_filenames is not None:
            filename = frame_filenames[i]
        else:
            filename = f"frame_{idx:06d}.jpg"
        filepath = frame_dir / filename
        frame = cv2.imread(str(filepath))
        if frame is None:
            logger.warning("Failed to read frame: %s", filepath)
            continue
        frames.append(frame)

    logger.info(
        "Loaded %d frames from %s",
        len(frames),
        frame_dir.name,
    )
    return frames, frame_indices


def read_video_frames(
    video_path: Path, max_frames: int | None = None, sample_rate: int = 1
) -> tuple[list[np.ndarray], list[int]]:
    """Read frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to read (None = all).
        sample_rate: Read every Nth frame.

    Returns:
        Tuple of (frames, frame_indices).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return [], []

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frames.append(frame)
                frame_indices.append(frame_idx)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    logger.info(
        "Read %d frames from %s (total frames: %d)",
        len(frames),
        video_path.name,
        frame_idx,
    )
    return frames, frame_indices


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    raw_config = load_config(args.config)

    # sam3.yaml wraps settings under a "sam3:" key; unwrap if present
    config = raw_config.get("sam3", raw_config)

    # Determine input mode: --frames-dir (recommended) or --video-dir (legacy)
    use_frames_dir = args.frames_dir is not None
    if not use_frames_dir and args.video_dir is None:
        logger.error("Either --frames-dir or --video-dir must be specified.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover inputs
    if use_frames_dir:
        inputs = discover_frame_dirs(args.frames_dir)
    else:
        inputs = discover_videos(args.video_dir)

    if not inputs:
        logger.error("No inputs found. Exiting.")
        return

    # Shard for multi-GPU parallel processing
    if args.shard_id is not None and args.num_shards is not None:
        inputs = [v for i, v in enumerate(inputs) if i % args.num_shards == args.shard_id]
        logger.info("Shard %d/%d: processing %d inputs", args.shard_id, args.num_shards, len(inputs))

    # Initialize tracker
    model_size = config.get("model_size", "large")
    device = config.get("device", "cuda")
    concept_prompts = config.get("concept_prompts", ["person", "object"])
    temporal_window = config.get("temporal_window", 16)
    sample_rate = config.get("sample_rate", 1)
    max_frames = config.get("max_frames", None)
    image_size_cfg = config.get("image_size", None)

    tracker = SAM3Tracker(model_size=model_size, device=device)
    tracker.set_concept_prompts(concept_prompts)

    # Process inputs
    for input_path in tqdm(inputs, desc="Processing"):
        video_id = input_path.stem if not use_frames_dir else input_path.name
        output_path = output_dir / f"{video_id}.pt"

        # Resume: skip already processed
        if args.resume and output_path.exists():
            logger.info("Skipping already-processed: %s", video_id)
            continue

        logger.info("Processing: %s", input_path)

        # Load frames
        if use_frames_dir:
            frames, frame_indices = load_extracted_frames(input_path)
        else:
            frames, frame_indices = read_video_frames(
                input_path, max_frames=max_frames, sample_rate=sample_rate
            )

        if not frames:
            logger.warning("No frames loaded from %s, skipping.", input_path)
            continue

        # Determine image_size from config or actual frame dimensions
        if image_size_cfg is not None:
            image_size = tuple(image_size_cfg)
        else:
            actual_h, actual_w = frames[0].shape[:2]
            image_size = (actual_h, actual_w)
            logger.info("Using actual frame size: %dx%d", actual_w, actual_h)

        extractor = FeatureExtractor(
            temporal_window=temporal_window,
            normalize_coords=True,
            image_size=image_size,
        )

        # Run SAM 3 tracking
        try:
            tracking_results = tracker.track_video(frames, frame_indices)
        except RuntimeError as e:
            logger.error("Tracking failed for %s: %s", video_id, e)
            continue

        # Extract features
        object_features, pairwise_features = extractor.extract(tracking_results)

        # Save results
        save_data = {
            "video_id": video_id,
            "video_path": str(input_path),
            "frame_indices": frame_indices,
            "tracking_results": tracking_results,
            "object_features": object_features,
            "pairwise_features": pairwise_features,
        }
        torch.save(save_data, output_path)
        logger.info("Saved tracking output to %s", output_path)

    logger.info("Done. Processed outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
