"""CLI inference entry point for event graph generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.inference.pipeline import InferencePipeline
from event_graph_generation.models.event_decoder import EventDecoder
from event_graph_generation.tracking.feature_extractor import FeatureExtractor
from event_graph_generation.tracking.sam3_tracker import SAM3Tracker
from event_graph_generation.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_action_names(actions_config_path: str) -> list[str]:
    """Load action names from a YAML config file.

    actions.yaml format is a list of dicts:
        actions:
          - name: "take_out"
          - name: "put_in"
    Returns: ["take_out", "put_in", ...]
    """
    path = Path(actions_config_path)
    if not path.exists():
        logger.warning("Actions config not found: %s, using empty action list", actions_config_path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    actions = data.get("actions", [])
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        return [entry["name"] for entry in actions if "name" in entry]
    return [str(a) for a in actions]


def build_event_decoder(config: dict, checkpoint_path: str, device: str) -> EventDecoder:
    """Build and load EventDecoder from checkpoint."""
    model_config = config.get("model", {})
    decoder = EventDecoder(
        d_model=model_config.get("d_model", 256),
        nhead=model_config.get("nhead", 8),
        num_object_encoder_layers=model_config.get("num_object_encoder_layers", 3),
        num_context_encoder_layers=model_config.get("num_context_encoder_layers", 3),
        num_event_decoder_layers=model_config.get("num_event_decoder_layers", 4),
        num_event_queries=model_config.get("num_event_queries", 20),
        max_objects=model_config.get("max_objects", 30),
        dropout=model_config.get("dropout", 0.1),
        d_geo=model_config.get("d_geo", 12),
        d_pair=model_config.get("d_pair", 7),
        num_actions=model_config.get("num_actions", 13),
        embedding_dim=model_config.get("embedding_dim", 256),
        temporal_window=model_config.get("temporal_window", 16),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["state_dict"])
    else:
        decoder.load_state_dict(checkpoint)

    decoder.to(device)
    decoder.eval()
    logger.info("Loaded EventDecoder from %s", checkpoint_path)
    return decoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run event graph inference on a video"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained EventDecoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/event_graph.json",
        help="Output path for the event graph JSON",
    )
    parser.add_argument(
        "--concept-prompts",
        nargs="+",
        default=None,
        help="Object category prompts for SAM3 tracking",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for events",
    )
    parser.add_argument(
        "--actions-config",
        type=str,
        default="configs/actions.yaml",
        help="Path to actions vocabulary YAML",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    # Load action names
    action_names = load_action_names(args.actions_config)
    config["action_names"] = action_names

    # Concept prompts: CLI args override config
    concept_prompts = args.concept_prompts or config.get(
        "concept_prompts",
        ["person", "wrench", "screwdriver", "drawer", "workbench"],
    )

    # Build components
    sam3_config = config.get("sam3", {})
    sam3_tracker = SAM3Tracker(
        model_size=sam3_config.get("model_size", "large"),
        device=device,
    )

    feature_config = config.get("feature_extractor", {})
    feature_extractor = FeatureExtractor(
        temporal_window=feature_config.get("temporal_window", 16),
        normalize_coords=feature_config.get("normalize_coords", True),
        image_size=tuple(feature_config.get("image_size", [480, 640])),
    )

    event_decoder = build_event_decoder(config, args.checkpoint, device)

    sampler_config = config.get("frame_sampler", {})
    frame_sampler = FrameSampler(
        target_fps=sampler_config.get("target_fps", 1.0),
    )

    # Create pipeline
    pipeline = InferencePipeline(
        sam3_tracker=sam3_tracker,
        feature_extractor=feature_extractor,
        event_decoder=event_decoder,
        frame_sampler=frame_sampler,
        config=config,
    )

    # Process video
    logger.info("Processing video: %s", args.video)
    event_graph = pipeline.process_video(
        video_path=args.video,
        concept_prompts=concept_prompts,
        confidence_threshold=args.confidence_threshold,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_graph.to_json(str(output_path))
    logger.info("Event graph saved to %s", output_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Event Graph Generation Complete")
    print(f"{'=' * 60}")
    print(f"Video:       {args.video}")
    print(f"Objects:     {len(event_graph.nodes)}")
    print(f"Events:      {len(event_graph.edges)}")
    print(f"Output:      {output_path}")
    print(f"{'=' * 60}")

    for node in event_graph.nodes:
        print(f"  [Object] track_id={node.track_id} category={node.category} "
              f"frames={node.first_seen_frame}-{node.last_seen_frame}")

    for edge in event_graph.edges:
        print(f"  [Event]  {edge.event_id}: agent={edge.agent_track_id} "
              f"--{edge.action}--> target={edge.target_track_id} "
              f"frame={edge.frame} conf={edge.confidence:.3f}")


if __name__ == "__main__":
    main()
