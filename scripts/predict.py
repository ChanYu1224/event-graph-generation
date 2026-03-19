"""Inference/prediction entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.models.base import build_model
from event_graph_generation.utils.io import load_checkpoint
from event_graph_generation.utils.logging import setup_logging


def _load_action_names(actions_config_path: str = "configs/actions.yaml") -> list[str]:
    """Load action names from actions.yaml."""
    path = Path(actions_config_path)
    if not path.exists():
        return []
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    actions = data.get("actions", [])
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        return [entry["name"] for entry in actions if "name" in entry]
    return [str(a) for a in actions]


def _run_event_decoder_inference(config: Config, args: argparse.Namespace) -> None:
    """Run inference using the InferencePipeline for event_decoder models."""
    from event_graph_generation.data.frame_sampler import FrameSampler
    from event_graph_generation.inference.pipeline import InferencePipeline
    from event_graph_generation.models.event_decoder import EventDecoder
    from event_graph_generation.tracking.feature_extractor import FeatureExtractor
    from event_graph_generation.tracking.sam3_tracker import SAM3Tracker

    device = config.training.device

    # Build EventDecoder
    decoder = EventDecoder(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_object_encoder_layers=config.model.num_object_encoder_layers,
        num_context_encoder_layers=config.model.num_context_encoder_layers,
        num_event_decoder_layers=config.model.num_event_decoder_layers,
        num_event_queries=config.model.num_event_queries,
        max_objects=config.model.max_objects,
        dropout=config.model.dropout,
        d_geo=config.model.d_geo,
        d_pair=config.model.d_pair,
        num_actions=config.model.num_actions,
        embedding_dim=config.model.embedding_dim,
    )
    load_checkpoint(args.checkpoint, decoder, device=device)
    decoder.to(device)
    decoder.eval()

    # Build components
    sam3_tracker = SAM3Tracker(
        model_size=config.sam3.model_size,
        device=device,
    )
    feature_extractor = FeatureExtractor(
        temporal_window=config.inference.clip_length,
    )
    frame_sampler = FrameSampler(target_fps=config.inference.target_fps)

    # Load action names from actions.yaml
    action_names = _load_action_names()

    # Build inference config dict
    inference_config = {
        "device": device,
        "clip_length": config.inference.clip_length,
        "clip_stride": config.inference.clip_stride,
        "max_objects": config.model.max_objects,
        "d_geo": config.model.d_geo,
        "d_pair": config.model.d_pair,
        "action_names": action_names,
        "dedup_frame_threshold": config.inference.nms_frame_threshold,
    }

    pipeline = InferencePipeline(
        sam3_tracker=sam3_tracker,
        feature_extractor=feature_extractor,
        event_decoder=decoder,
        frame_sampler=frame_sampler,
        config=inference_config,
    )

    concept_prompts = list(config.sam3.concept_prompts)

    event_graph = pipeline.process_video(
        video_path=args.input,
        concept_prompts=concept_prompts,
    )

    # Save as JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".json"):
        event_graph.to_json(str(output_path))
    else:
        # Save as .pt for backwards compat
        torch.save(event_graph.to_dict(), args.output)

    print(f"Event graph saved to {args.output} "
          f"({len(event_graph.nodes)} objects, {len(event_graph.edges)} events)")


def _run_legacy_inference(config: Config, args: argparse.Namespace) -> None:
    """Run legacy inference path for non-event_decoder models."""
    model = build_model(config.model)
    load_checkpoint(args.checkpoint, model, device=config.training.device)
    model.eval()

    # TODO: Load input data and run inference
    # predictions = model(input_data)
    # torch.save(predictions, args.output)

    print(f"Predictions saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, default="predictions.pt", help="Output path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    setup_logging()

    if config.model.name == "event_decoder":
        _run_event_decoder_inference(config, args)
    else:
        _run_legacy_inference(config, args)


if __name__ == "__main__":
    main()
