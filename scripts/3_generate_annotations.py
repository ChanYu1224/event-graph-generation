"""Batch VLM annotation script for video event extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from event_graph_generation.annotation.validator import AnnotationValidator
from event_graph_generation.annotation.vlm_annotator import VLMAnnotator

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch VLM annotation for video event extraction"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vlm.yaml"),
        help="VLM config YAML (default: configs/vlm.yaml)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory containing video files (default: data/videos)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Output directory for annotation JSONs (default: data/annotations)",
    )
    parser.add_argument(
        "--actions-config",
        type=Path,
        default=Path("configs/actions.yaml"),
        help="Actions vocabulary YAML (default: configs/actions.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed videos",
    )
    return parser.parse_args()


def load_vlm_config(config_path: Path) -> dict:
    """Load VLM configuration from YAML."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("vlm", cfg)


def load_actions_config(actions_path: Path) -> tuple[list[str], list[dict]]:
    """Load action vocabulary and config from YAML.

    Returns:
        Tuple of (action_names, action_config_list).
    """
    with open(actions_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    action_entries = cfg.get("actions", [])
    action_names = [entry["name"] for entry in action_entries]
    return action_names, action_entries


def discover_videos(video_dir: Path) -> list[Path]:
    """Find all video files in the directory."""
    if not video_dir.exists():
        return []
    return sorted(
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def get_processed_videos(output_dir: Path) -> set[str]:
    """Get set of already-processed video IDs from output directory."""
    processed = set()
    if not output_dir.exists():
        return processed
    for p in output_dir.iterdir():
        if p.suffix == ".json":
            processed.add(p.stem)
    return processed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    # Load configs
    vlm_cfg = load_vlm_config(args.config)
    action_names, action_entries = load_actions_config(args.actions_config)

    logger.info("Loaded %d actions from %s", len(action_names), args.actions_config)

    # Extract categories from actions config (not specified separately, use empty for now)
    # Categories are discovered dynamically by the VLM; we pass an open list
    categories: list[str] = vlm_cfg.get("categories", [])

    # Discover videos
    videos = discover_videos(args.video_dir)
    if not videos:
        logger.error("No videos found in %s", args.video_dir)
        return

    logger.info("Found %d videos in %s", len(videos), args.video_dir)

    # Resume: skip already-processed
    if args.resume:
        processed = get_processed_videos(args.output_dir)
        before = len(videos)
        videos = [v for v in videos if v.stem not in processed]
        logger.info(
            "Resume: skipped %d already-processed, %d remaining",
            before - len(videos),
            len(videos),
        )
        if not videos:
            logger.info("All videos already processed")
            return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize annotator
    annotator = VLMAnnotator(
        model_name=vlm_cfg.get("model_name", "Qwen/Qwen3.5-9B"),
        device_map=vlm_cfg.get("device_map", "auto"),
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
        max_new_tokens=vlm_cfg.get("max_new_tokens", 4096),
        temperature=vlm_cfg.get("temperature", 0.1),
        thinking=vlm_cfg.get("thinking", False),
    )

    # Initialize validator
    validator = AnnotationValidator(
        allowed_actions=action_names,
        allowed_categories=categories,
        action_config=action_entries,
    )

    # Processing settings
    clip_length = vlm_cfg.get("clip_length", 16)
    clip_stride = vlm_cfg.get("clip_stride", 8)
    target_fps = vlm_cfg.get("target_fps", 1.0)

    # Stats
    total_success = 0
    total_errors = 0
    total_clips = 0
    all_annotations = []

    for video_path in tqdm(videos, desc="Annotating videos"):
        video_id = video_path.stem
        output_path = args.output_dir / f"{video_id}.json"

        try:
            annotations = annotator.annotate_video(
                video_path=str(video_path),
                fps=target_fps,
                clip_length=clip_length,
                clip_stride=clip_stride,
                categories=categories,
                actions=action_names,
            )

            # Validate all clip annotations
            validated, stats = validator.validate_batch(annotations)
            all_annotations.extend(validated)
            total_clips += len(annotations)

            # Save as JSON
            output_data = {
                "video_id": video_id,
                "video_path": str(video_path),
                "num_clips": len(validated),
                "validation_stats": stats,
                "clips": [ann.model_dump() for ann in validated],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            total_success += 1
            logger.info(
                "Saved %d clips for %s (discard_rate=%.2f%%)",
                len(validated),
                video_id,
                stats["discard_rate"] * 100,
            )

        except Exception as e:
            total_errors += 1
            logger.error("Failed to process %s: %s", video_id, e)
            continue

    # Summary
    logger.info("=" * 50)
    logger.info("Annotation complete")
    logger.info("Videos processed: %d success, %d errors", total_success, total_errors)
    logger.info("Total clips annotated: %d", total_clips)


if __name__ == "__main__":
    main()
