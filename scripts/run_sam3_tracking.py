"""Batch SAM 3 tracking script with resume support.

Usage:
    python scripts/run_sam3_tracking.py --config configs/sam3.yaml --video-dir data/videos --output-dir data/sam3_outputs --resume
"""

from __future__ import annotations

import argparse
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
        default="data/videos",
        help="Directory containing video files.",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    videos = discover_videos(args.video_dir)
    if not videos:
        logger.error("No video files found. Exiting.")
        return

    # Shard videos for multi-GPU parallel processing
    if args.shard_id is not None and args.num_shards is not None:
        videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_id]
        logger.info("Shard %d/%d: processing %d videos", args.shard_id, args.num_shards, len(videos))

    # Initialize tracker
    model_size = config.get("model_size", "large")
    device = config.get("device", "cuda")
    concept_prompts = config.get("concept_prompts", ["person", "object"])
    temporal_window = config.get("temporal_window", 16)
    sample_rate = config.get("sample_rate", 1)
    max_frames = config.get("max_frames", None)
    image_size = tuple(config.get("image_size", [480, 640]))

    tracker = SAM3Tracker(model_size=model_size, device=device)
    tracker.set_concept_prompts(concept_prompts)

    extractor = FeatureExtractor(
        temporal_window=temporal_window,
        normalize_coords=True,
        image_size=image_size,
    )

    # Process videos
    for video_path in tqdm(videos, desc="Processing videos"):
        video_id = video_path.stem
        output_path = output_dir / f"{video_id}.pt"

        # Resume: skip already processed
        if args.resume and output_path.exists():
            logger.info("Skipping already-processed video: %s", video_id)
            continue

        logger.info("Processing video: %s", video_path)

        # Read frames
        frames, frame_indices = read_video_frames(
            video_path, max_frames=max_frames, sample_rate=sample_rate
        )
        if not frames:
            logger.warning("No frames read from %s, skipping.", video_path)
            continue

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
            "video_path": str(video_path),
            "frame_indices": frame_indices,
            "tracking_results": tracking_results,
            "object_features": object_features,
            "pairwise_features": pairwise_features,
        }
        torch.save(save_data, output_path)
        logger.info("Saved tracking output to %s", output_path)

    logger.info("Done. Processed videos saved to %s", output_dir)


if __name__ == "__main__":
    main()
