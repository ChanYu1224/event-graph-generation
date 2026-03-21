"""Backfill timestamp metadata into existing annotation JSONs.

Reads annotation JSONs from ``--annotations-dir``, enriches each with
``video_metadata``, ``coverage``, and per-clip ``clip_metadata``, then
writes the enriched files to ``--output-dir`` (leaving originals untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.utils.timestamps import (
    compute_clip_timestamps,
    enrich_clips,
    offset_to_iso,
    parse_video_start_time,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich annotation JSONs with timestamp metadata"
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Directory containing annotation JSONs (default: data/annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations_enriched"),
        help="Output directory for enriched JSONs (default: data/annotations_enriched)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/resized/room"),
        help="Directory containing video files for FPS lookup (default: data/resized/room)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        help="Sampling FPS used during annotation (default: 1.0)",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="Number of sampled frames per clip (default: 16)",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=8,
        help="Stride in sampled-frame space between clips (default: 8)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional VLM config YAML to read clip_length/clip_stride/target_fps",
    )
    return parser.parse_args()


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _find_video_file(video_dir: Path, video_id: str) -> Path | None:
    """Find video file for a given video ID, checking all supported extensions.

    Args:
        video_dir: Directory containing video files.
        video_id: Video identifier (filename stem).

    Returns:
        Path to the video file, or ``None`` if not found.
    """
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def enrich_annotation(
    annotation: dict,
    video_path: Path,
    target_fps: float,
    clip_length: int,
    clip_stride: int,
) -> dict:
    """Add timestamp metadata to an annotation dict.

    Args:
        annotation: Original annotation dict loaded from JSON.
        video_path: Path to the corresponding video file.
        target_fps: Sampling FPS used during annotation.
        clip_length: Number of sampled frames per clip.
        clip_stride: Stride in sampled-frame space between clips.

    Returns:
        New dict with ``video_metadata``, ``coverage``, and per-clip
        ``clip_metadata`` added.
    """
    video_id = annotation["video_id"]
    video_start_time = parse_video_start_time(video_id)

    video_info = FrameSampler.get_video_info(video_path)
    source_fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    duration_sec = video_info["duration_sec"]

    clip_timestamps = compute_clip_timestamps(
        source_fps=source_fps,
        target_fps=target_fps,
        total_frames=total_frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        video_start_time=video_start_time,
    )

    clips = annotation.get("clips", [])
    enrichment = enrich_clips(clips, clip_timestamps)

    video_start_iso = video_start_time.strftime("%Y-%m-%dT%H:%M:%S")
    video_end_iso = offset_to_iso(video_start_time, duration_sec)

    return {
        "video_id": video_id,
        "video_path": annotation.get("video_path", str(video_path)),
        "video_metadata": {
            "video_start_time": video_start_iso,
            "source_fps": round(source_fps, 2),
            "target_fps": target_fps,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 1),
            "video_end_time": video_end_iso,
        },
        "coverage": {
            "total_clips": len(clips),
            "annotated_clips": enrichment.annotated_clips,
            "motion_filtered_clips": enrichment.motion_filtered_clips,
            "clips_with_events": enrichment.clips_with_events,
            "time_range": f"{video_start_iso} ~ {video_end_iso}",
            "motion_filtered_ranges": enrichment.motion_filtered_ranges,
        },
        "num_clips": annotation.get("num_clips", len(clips)),
        "validation_stats": annotation.get("validation_stats", {}),
        "clips": enrichment.clips,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    # Override from config YAML if provided
    if args.config is not None:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vlm_cfg = cfg.get("vlm", cfg)
        args.target_fps = vlm_cfg.get("target_fps", args.target_fps)
        args.clip_length = vlm_cfg.get("clip_length", args.clip_length)
        args.clip_stride = vlm_cfg.get("clip_stride", args.clip_stride)

    # Discover annotation JSONs
    if not args.annotations_dir.exists():
        logger.error("Annotations directory not found: %s", args.annotations_dir)
        return

    annotation_files = sorted(args.annotations_dir.glob("*.json"))
    if not annotation_files:
        logger.error("No annotation JSONs found in %s", args.annotations_dir)
        return

    logger.info("Found %d annotation files in %s", len(annotation_files), args.annotations_dir)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0

    for ann_path in annotation_files:
        video_id = ann_path.stem
        video_path = _find_video_file(args.video_dir, video_id)

        if video_path is None:
            logger.warning("Video not found for %s in %s", video_id, args.video_dir)
            errors += 1
            continue

        try:
            with open(ann_path, encoding="utf-8") as f:
                annotation = json.load(f)

            enriched = enrich_annotation(
                annotation=annotation,
                video_path=video_path,
                target_fps=args.target_fps,
                clip_length=args.clip_length,
                clip_stride=args.clip_stride,
            )

            output_path = args.output_dir / f"{video_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)

            coverage = enriched["coverage"]
            logger.info(
                "Enriched %s: %d clips (%d annotated, %d motion-filtered, %d with events)",
                video_id,
                coverage["total_clips"],
                coverage["annotated_clips"],
                coverage["motion_filtered_clips"],
                coverage["clips_with_events"],
            )
            success += 1

        except Exception as e:
            logger.error("Failed to enrich %s: %s", video_id, e)
            errors += 1
            continue

    logger.info("=" * 50)
    logger.info("Enrichment complete: %d success, %d errors", success, errors)


if __name__ == "__main__":
    main()
