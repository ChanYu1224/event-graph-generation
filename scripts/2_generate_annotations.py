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
from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.utils.timestamps import (
    compute_clip_timestamps,
    enrich_clips,
    offset_to_iso,
    parse_video_start_time,
)

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
        "--vocab-config",
        type=Path,
        default=Path("configs/vocab.yaml"),
        help="Categories and attribute vocabulary YAML (default: configs/vocab.yaml)",
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


def load_vocab_config(
    vocab_path: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    """Load categories and attribute vocabulary from YAML.

    Args:
        vocab_path: Path to the vocabulary YAML file.

    Returns:
        Tuple of (categories, attribute_vocab).
    """
    with open(vocab_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    categories: list[str] = cfg.get("categories", [])
    attribute_vocab: dict[str, list[str]] = cfg.get("attribute_vocab", {})
    return categories, attribute_vocab


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

    # Load categories and attribute vocabulary from vocab config
    categories, attribute_vocab = load_vocab_config(args.vocab_config)
    logger.info(
        "Loaded %d categories and %d attribute axes from %s",
        len(categories),
        len(attribute_vocab),
        args.vocab_config,
    )

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
        backend=vlm_cfg.get("backend", "transformers"),
        tensor_parallel_size=vlm_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vlm_cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=vlm_cfg.get("max_model_len", 32768),
        max_num_seqs=vlm_cfg.get("max_num_seqs", 5),
        limit_mm_per_prompt=vlm_cfg.get("limit_mm_per_prompt", 16),
        api_base=vlm_cfg.get("api_base", "http://localhost:8000/v1"),
        max_concurrent_requests=vlm_cfg.get("max_concurrent_requests", 8),
    )

    # Initialize validator
    validator = AnnotationValidator(
        allowed_actions=action_names,
        allowed_categories=categories,
        action_config=action_entries,
        attribute_vocab=attribute_vocab,
    )

    # Processing settings
    clip_length = vlm_cfg.get("clip_length", 16)
    clip_stride = vlm_cfg.get("clip_stride", 8)
    target_fps = vlm_cfg.get("target_fps", 1.0)
    motion_filter_enabled = vlm_cfg.get("motion_filter_enabled", False)
    motion_threshold = vlm_cfg.get("motion_threshold", 3.0)

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
                attribute_vocab=attribute_vocab,
                motion_filter_enabled=motion_filter_enabled,
                motion_threshold=motion_threshold,
            )

            # Validate all clip annotations
            validated, stats = validator.validate_batch(annotations)
            all_annotations.extend(validated)
            total_clips += len(annotations)

            # Compute timestamp metadata
            video_info = FrameSampler.get_video_info(video_path)
            source_fps = video_info["fps"]
            total_frames = video_info["total_frames"]
            duration_sec = video_info["duration_sec"]

            video_start_time = parse_video_start_time(video_id)
            video_start_iso = video_start_time.strftime("%Y-%m-%dT%H:%M:%S")
            video_end_iso = offset_to_iso(video_start_time, duration_sec)

            clip_timestamps = compute_clip_timestamps(
                source_fps=source_fps,
                target_fps=target_fps,
                total_frames=total_frames,
                clip_length=clip_length,
                clip_stride=clip_stride,
                video_start_time=video_start_time,
            )

            # Enrich clips with metadata
            clip_dicts = [ann.model_dump() for ann in validated]
            enrichment = enrich_clips(clip_dicts, clip_timestamps)

            # Save as JSON
            output_data = {
                "video_id": video_id,
                "video_path": str(video_path),
                "video_metadata": {
                    "video_start_time": video_start_iso,
                    "source_fps": round(source_fps, 2),
                    "target_fps": target_fps,
                    "total_frames": total_frames,
                    "duration_sec": round(duration_sec, 1),
                    "video_end_time": video_end_iso,
                },
                "coverage": {
                    "total_clips": len(validated),
                    "annotated_clips": enrichment.annotated_clips,
                    "motion_filtered_clips": enrichment.motion_filtered_clips,
                    "clips_with_events": enrichment.clips_with_events,
                    "time_range": f"{video_start_iso} ~ {video_end_iso}",
                    "motion_filtered_ranges": enrichment.motion_filtered_ranges,
                },
                "num_clips": len(validated),
                "validation_stats": stats,
                "clips": enrichment.clips,
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
