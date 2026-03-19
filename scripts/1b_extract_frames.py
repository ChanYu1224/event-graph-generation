"""Extract frames from videos at a target FPS and save as JPEG files.

Saves frames as JPEG images with metadata, enabling memory-efficient
downstream processing (SAM3 tracking, VLM annotation) without loading
entire videos into RAM.

Usage:
    uv run python scripts/1b_extract_frames.py \
        --video-dir data/resized/room \
        --output-dir data/frames/room \
        --fps 5.0 \
        --quality 95 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos at a target FPS.")
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frames per second for extraction (default: 5.0).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100, default: 95).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that already have metadata.json in output dir.",
    )
    return parser.parse_args()


def discover_videos(video_dir: str) -> list[Path]:
    """Find all video files in the given directory (recursive)."""
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        logger.error("Video directory does not exist: %s", video_dir)
        return []

    extensions = {".mp4", ".avi", ".mov"}
    videos = sorted(p for p in video_dir_path.rglob("*") if p.suffix.lower() in extensions)
    logger.info("Discovered %d video files in %s", len(videos), video_dir)
    return videos


def iter_sampled_frames(video_path: Path, target_fps: float):
    """Yield (frame, frame_index) one at a time at the target FPS.

    Uses cv2.CAP_PROP_POS_FRAMES to seek directly, avoiding decoding
    of intermediate frames and keeping memory usage at ~1 frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot determine FPS for {video_path}")

    interval = max(1, round(source_fps / target_fps))

    try:
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, frame_idx, source_fps, total_frames, interval
            frame_idx += interval
    finally:
        cap.release()


def parse_video_start_time(video_stem: str) -> datetime | None:
    """Parse start datetime from video filename like YYYYMMDD_HHMMSS_tpNNNNN."""
    try:
        return datetime.strptime(video_stem[:15], "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        return None


def make_frame_filename(start_time: datetime, frame_idx: int, source_fps: float) -> str:
    """Generate datetime-based filename: YYYYMMDD_HHMMSS_mmm.jpg"""
    timestamp_sec = frame_idx / source_fps
    frame_time = start_time + timedelta(seconds=timestamp_sec)
    ms = frame_time.microsecond // 1000
    return frame_time.strftime("%Y%m%d_%H%M%S") + f"_{ms:03d}.jpg"


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    quality: int,
) -> None:
    """Extract frames from a single video and save as JPEG + metadata."""
    video_id = video_path.stem
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = parse_video_start_time(video_id)
    if start_time is None:
        raise ValueError(
            f"Cannot parse start time from video filename: {video_id}. "
            "Expected format: YYYYMMDD_HHMMSS_..."
        )

    frame_indices: list[int] = []
    frame_filenames: list[str] = []
    source_fps = 0.0
    total_frames = 0
    interval = 0
    num_saved = 0

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    for frame, frame_idx, source_fps, total_frames, interval in iter_sampled_frames(
        video_path, target_fps
    ):
        filename = make_frame_filename(start_time, frame_idx, source_fps)
        filepath = video_output_dir / filename
        cv2.imwrite(str(filepath), frame, encode_params)
        frame_indices.append(frame_idx)
        frame_filenames.append(filename)
        num_saved += 1

    if num_saved == 0:
        logger.warning("No frames extracted from %s", video_path)
        return

    metadata = {
        "video_id": video_id,
        "source_video": str(video_path),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "frame_interval": interval,
        "total_source_frames": total_frames,
        "num_extracted_frames": num_saved,
        "frame_indices": frame_indices,
        "frame_filenames": frame_filenames,
    }

    metadata_path = video_output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Extracted %d frames from %s (source_fps=%.1f, interval=%d) -> %s",
        num_saved,
        video_path.name,
        source_fps,
        interval,
        video_output_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args.video_dir)
    if not videos:
        logger.error("No video files found. Exiting.")
        return

    skipped = 0
    processed = 0

    for video_path in videos:
        video_id = video_path.stem
        metadata_path = output_dir / video_id / "metadata.json"

        if args.resume and metadata_path.exists():
            logger.info("Skipping already-extracted video: %s", video_id)
            skipped += 1
            continue

        extract_video_frames(video_path, output_dir, args.fps, args.quality)
        processed += 1

    logger.info(
        "Done. Processed: %d, Skipped: %d, Total: %d",
        processed,
        skipped,
        len(videos),
    )


if __name__ == "__main__":
    main()
