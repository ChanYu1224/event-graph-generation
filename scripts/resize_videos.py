"""Resize videos for SAM3 processing (long side → max_side px).

Reads 2K videos from input directory, resizes so the long side equals
max_side (default 1008, SAM3 internal resolution), and writes to output
directory preserving subdirectory structure.

Usage:
    uv run python scripts/resize_videos.py --input-dir data/mp4 --output-dir data/resized --max-side 1008
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize videos for SAM3 processing.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/mp4",
        help="Directory containing original video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/resized",
        help="Directory to save resized videos.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1008,
        help="Target long-side resolution in pixels.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed videos.",
    )
    return parser.parse_args()


def discover_videos(video_dir: Path) -> list[Path]:
    """Find all video files recursively."""
    if not video_dir.exists():
        logger.error("Input directory does not exist: %s", video_dir)
        return []
    videos = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    logger.info("Discovered %d video files in %s", len(videos), video_dir)
    return videos


def resize_video(
    input_path: Path,
    output_path: Path,
    max_side: int,
) -> bool:
    """Resize a single video so that its long side equals max_side.

    Returns True on success, False on failure.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", input_path)
        return False

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute new dimensions (preserve aspect ratio, long side = max_side)
    scale = max_side / max(orig_h, orig_w)
    new_w = int(orig_w * scale) & ~1  # ensure even (codec requirement)
    new_h = int(orig_h * scale) & ~1

    logger.info(
        "%s: %dx%d → %dx%d (scale=%.3f, %d frames)",
        input_path.name,
        orig_w,
        orig_h,
        new_w,
        new_h,
        scale,
        total_frames,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))
    if not writer.isOpened():
        logger.error("Failed to create VideoWriter for: %s", output_path)
        cap.release()
        return False

    try:
        for _ in tqdm(
            range(total_frames),
            desc=input_path.name,
            leave=False,
            unit="frame",
        ):
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()

    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    videos = discover_videos(input_dir)
    if not videos:
        logger.error("No video files found. Exiting.")
        return

    success = 0
    skipped = 0

    for video_path in tqdm(videos, desc="Resizing videos"):
        # Preserve subdirectory structure (e.g. kitchen/video.mp4)
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path

        if args.resume and output_path.exists():
            logger.info("Skipping already-processed: %s", rel_path)
            skipped += 1
            continue

        if resize_video(video_path, output_path, args.max_side):
            success += 1
        else:
            logger.error("Failed to resize: %s", video_path)

    logger.info(
        "Done. Resized: %d, Skipped: %d, Total: %d",
        success,
        skipped,
        len(videos),
    )


if __name__ == "__main__":
    main()
