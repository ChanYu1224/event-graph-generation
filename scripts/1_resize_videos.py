"""Resize videos for SAM3 processing (long side → max_side px).

Reads 2K videos from input directory, resizes so the long side equals
max_side (default 1008, SAM3 internal resolution), and writes to output
directory preserving subdirectory structure.

Uses ffmpeg for fast, hardware-friendly transcoding instead of per-frame
OpenCV loops.

Usage:
    uv run python scripts/1_resize_videos.py --input-dir data/raw --output-dir data/resized --max-side 1008
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    parser.add_argument(
        "--no-hwaccel",
        action="store_true",
        help="Disable CUDA hardware acceleration (NVDEC/NVENC).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel ffmpeg processes.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for round-robin assignment (e.g. '0,1'). "
        "Defaults to CUDA_VISIBLE_DEVICES, or '0' if unset.",
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


def _check_nvenc_available() -> bool:
    """Check if NVIDIA NVENC/NVDEC hardware acceleration is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False


def resize_video(
    input_path: Path,
    output_path: Path,
    max_side: int,
    *,
    hwaccel: bool = False,
    gpu_id: int = 0,
) -> bool:
    """Resize a single video using ffmpeg so that its long side equals max_side.

    When hwaccel=True, uses NVDEC for decoding, scale_cuda for GPU-side
    resizing, and NVENC for encoding — keeping the entire pipeline on GPU.
    gpu_id selects which GPU to use via -hwaccel_device.

    Returns True on success, False on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hwaccel:
        scale_filter = (
            f"scale_cuda='if(gte(iw,ih),{max_side},-2)':'if(gte(iw,ih),-2,{max_side})'"
        )
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(gpu_id),
            "-hwaccel_output_format", "cuda",
            "-i", str(input_path),
            "-vf", scale_filter,
            "-c:v", "h264_nvenc",
            "-cq", "30",
            "-preset", "p4",
            "-y",
            str(output_path),
        ]
    else:
        scale_filter = (
            f"scale='if(gte(iw,ih),{max_side},-2)':'if(gte(iw,ih),-2,{max_side})'"
        )
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-y",
            str(output_path),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg failed for %s (exit %d): %s",
                input_path.name,
                result.returncode,
                result.stderr[-500:] if result.stderr else "",
            )
            return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False

    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg is not installed or not on PATH. Exiting.")
        return

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    hwaccel = False
    if not args.no_hwaccel and _check_nvenc_available():
        hwaccel = True
        logger.info("CUDA hardware acceleration enabled (NVDEC + scale_cuda + NVENC)")
    else:
        logger.info("Using CPU encoding (libx264)")

    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        # CUDA_VISIBLE_DEVICES remaps physical GPUs to logical 0..N-1
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        n_gpus = len([x for x in cvd.split(",") if x.strip()])
        gpu_ids = list(range(n_gpus))
    logger.info("GPU IDs: %s, Workers: %d", gpu_ids, args.workers)

    videos = discover_videos(input_dir)
    if not videos:
        logger.error("No video files found. Exiting.")
        return

    # Build task list, filtering out already-processed videos upfront
    tasks: list[tuple[Path, Path]] = []
    skipped = 0
    for video_path in videos:
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        if args.resume and output_path.exists():
            logger.info("Skipping already-processed: %s", rel_path)
            skipped += 1
        else:
            tasks.append((video_path, output_path))

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, (video_path, output_path) in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            fut = executor.submit(
                resize_video, video_path, output_path, args.max_side,
                hwaccel=hwaccel, gpu_id=gpu_id,
            )
            futures[fut] = video_path

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Resizing videos"):
            video_path = futures[fut]
            if fut.result():
                success += 1
            else:
                failed += 1
                logger.error("Failed to resize: %s", video_path)

    logger.info(
        "Done. Resized: %d, Skipped: %d, Failed: %d, Total: %d",
        success,
        skipped,
        failed,
        len(videos),
    )


if __name__ == "__main__":
    main()
