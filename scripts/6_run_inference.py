"""Run inference on a video using trained V-JEPA event graph model.

End-to-end pipeline: video → V-JEPA feature extraction → event prediction → EventGraph JSON.

Usage:
    uv run python scripts/6_run_inference.py \
        --video data/resized/room/video.mp4 \
        --checkpoint checkpoints/vjepa_vitl_20260323_094236/best.pt \
        --config configs/vjepa_training.yaml \
        --vjepa-config configs/vjepa.yaml \
        --output output/event_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.inference.postprocess import (
    build_event_graph,
    predictions_to_events,
)
from event_graph_generation.models.base import build_model
from event_graph_generation.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_VJEPA_CORRECT_URL = "https://dl.fbaipublicfiles.com/vjepa2"


def _patch_vjepa_hub_url() -> None:
    """Patch V-JEPA 2 hub source to use the correct checkpoint URL."""
    hub_dir = Path(torch.hub.get_dir())
    backbones_path = (
        hub_dir / "facebookresearch_vjepa2_main" / "src" / "hub" / "backbones.py"
    )
    if not backbones_path.exists():
        return
    content = backbones_path.read_text()
    if "localhost:8300" in content:
        content = content.replace(
            'VJEPA_BASE_URL = "http://localhost:8300"',
            f'VJEPA_BASE_URL = "{_VJEPA_CORRECT_URL}"',
        )
        backbones_path.write_text(content)
        for key in list(sys.modules.keys()):
            if "facebookresearch_vjepa2" in key or "src.hub" in key:
                del sys.modules[key]
        logger.info("Patched V-JEPA hub URL")


def _bgr_to_pil(bgr_array: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB Image."""
    rgb = bgr_array[:, :, ::-1]
    return Image.fromarray(rgb)


def _load_vjepa_encoder(
    vjepa_config: object,
    device: str,
    dtype: torch.dtype,
) -> tuple:
    """Load V-JEPA encoder and preprocessor.

    Args:
        vjepa_config: VJEPAConfig dataclass.
        device: Device string.
        dtype: Model dtype.

    Returns:
        (encoder, preprocessor) tuple.
    """
    if vjepa_config.backend == "hub":
        logger.info(
            "Loading V-JEPA via PyTorch Hub: %s/%s",
            vjepa_config.hub_repo,
            vjepa_config.hub_model_name,
        )
        torch.hub.list(vjepa_config.hub_repo, force_reload=False)
        _patch_vjepa_hub_url()

        preprocessor = torch.hub.load(
            vjepa_config.hub_repo,
            "vjepa2_preprocessor",
            crop_size=vjepa_config.image_size,
        )
        encoder, _predictor = torch.hub.load(
            vjepa_config.hub_repo,
            vjepa_config.hub_model_name,
        )
        encoder = encoder.to(device=device, dtype=dtype).eval()
    else:
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading V-JEPA via HuggingFace: %s", vjepa_config.model_name)
        preprocessor = AutoProcessor.from_pretrained(vjepa_config.model_name)
        encoder = (
            AutoModel.from_pretrained(vjepa_config.model_name, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    return encoder, preprocessor


def _extract_clip_tokens(
    frames: list,
    encoder: torch.nn.Module,
    preprocessor: object,
    device: str,
    dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    """Extract V-JEPA tokens from a clip of frames.

    Args:
        frames: List of SampledFrame objects for this clip.
        encoder: V-JEPA encoder model.
        preprocessor: V-JEPA preprocessor.
        device: Device string.
        dtype: Model dtype.
        backend: "hub" or "hf".

    Returns:
        Tensor of shape (S, D) — spatiotemporal tokens.
    """
    if backend == "hub":
        images = [_bgr_to_pil(f.image) for f in frames]
        clip_tensor = preprocessor(images)
        clip_tensor = clip_tensor[0].unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast(device, dtype=dtype):
            tokens = encoder(clip_tensor).squeeze(0).float().cpu()
    else:
        images = [f.image[:, :, ::-1] for f in frames]
        inputs = preprocessor(videos=images, return_tensors="pt")
        pixel_values = inputs["pixel_values_videos"].to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = encoder(pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                tokens = outputs.last_hidden_state.squeeze(0).cpu()
            else:
                tokens = outputs[0].squeeze(0).cpu()

    return tokens


def _deduplicate_events(
    all_events: list[dict],
    frame_threshold: int = 3,
) -> list[dict]:
    """Remove duplicate events from overlapping clips via NMS-like dedup.

    Events with the same action, agent, and target within frame_threshold
    frames are considered duplicates. The higher-confidence one is kept.

    Args:
        all_events: All events from all clips.
        frame_threshold: Max frame distance to consider as duplicate.

    Returns:
        Deduplicated list of events.
    """
    if not all_events:
        return []

    # Sort by confidence descending
    sorted_events = sorted(all_events, key=lambda e: e["confidence"], reverse=True)
    kept: list[dict] = []

    for event in sorted_events:
        is_duplicate = False
        for existing in kept:
            if (
                event["action"] == existing["action"]
                and event["agent_track_id"] == existing["agent_track_id"]
                and event["target_track_id"] == existing["target_track_id"]
                and abs(event["frame"] - existing["frame"]) <= frame_threshold
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(event)

    return sorted(kept, key=lambda e: e["frame"])


def run_inference(
    video_path: Path,
    checkpoint_path: Path,
    config: Config,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    actions_config: str = "configs/actions.yaml",
    vocab_config: str = "configs/vocab.yaml",
) -> dict:
    """Run end-to-end inference on a single video.

    Args:
        video_path: Path to input video file.
        checkpoint_path: Path to model checkpoint (.pt).
        config: Merged Config object (training + vjepa).
        device: Device for inference.
        confidence_threshold: Minimum interaction score to keep an event.
        actions_config: Path to actions.yaml.
        vocab_config: Path to vocab.yaml.

    Returns:
        EventGraph as a dict.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.vjepa.dtype, torch.bfloat16)

    # Load action and category vocabularies
    with open(actions_config) as f:
        actions_data = yaml.safe_load(f)
    action_names = [a["name"] for a in actions_data["actions"]]

    with open(vocab_config) as f:
        vocab_data = yaml.safe_load(f)
    category_names = vocab_data["categories"] + ["unknown"]

    # Build and load event graph model
    logger.info("Building model...")
    model = build_model(config.model, vjepa_config=config.vjepa)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info(
        "Loaded checkpoint from %s (epoch %s)",
        checkpoint_path,
        ckpt.get("epoch", "?"),
    )

    # Load V-JEPA encoder
    vjepa_encoder, preprocessor = _load_vjepa_encoder(config.vjepa, device, dtype)

    # Sample frames from video
    clip_length = config.vjepa.frames_per_clip
    clip_stride = clip_length // 2
    sampler = FrameSampler(target_fps=1.0)
    frames = sampler.sample(video_path)
    logger.info("Sampled %d frames from %s", len(frames), video_path.name)

    if len(frames) < clip_length:
        logger.warning(
            "Video has only %d frames (need %d). Returning empty graph.",
            len(frames),
            clip_length,
        )
        return {"video_id": video_path.stem, "nodes": [], "edges": [], "metadata": {}}

    # Process each clip
    all_events: list[dict] = []
    all_object_info: dict[int, dict] = {}  # slot_id -> object info
    num_slots = config.model.object_pooling.num_slots

    for clip_start in range(0, len(frames) - clip_length + 1, clip_stride):
        clip_frames = frames[clip_start : clip_start + clip_length]
        frame_indices = [f.frame_index for f in clip_frames]

        # V-JEPA feature extraction
        tokens = _extract_clip_tokens(
            clip_frames,
            vjepa_encoder,
            preprocessor,
            device,
            dtype,
            config.vjepa.backend,
        )

        # Event graph prediction
        vjepa_tokens = tokens.unsqueeze(0).to(device)  # (1, S, D)
        with torch.no_grad():
            obj_repr, predictions = model(vjepa_tokens)

        # Derive object info from ObjectRepresentation
        existence = torch.sigmoid(obj_repr.existence[0]).cpu()  # (K,)
        categories = obj_repr.categories[0].cpu()  # (K, C)

        track_id_map = {}
        for slot_idx in range(num_slots):
            if float(existence[slot_idx]) < 0.3:
                continue
            track_id_map[slot_idx] = slot_idx
            cat_idx = int(torch.argmax(categories[slot_idx]))
            cat_name = (
                category_names[cat_idx]
                if cat_idx < len(category_names)
                else "unknown"
            )
            if slot_idx not in all_object_info:
                all_object_info[slot_idx] = {
                    "track_id": slot_idx,
                    "category": cat_name,
                    "first_seen_frame": frame_indices[0],
                    "last_seen_frame": frame_indices[-1],
                    "confidence": float(existence[slot_idx]),
                }
            else:
                obj = all_object_info[slot_idx]
                obj["last_seen_frame"] = max(
                    obj["last_seen_frame"], frame_indices[-1]
                )
                obj["confidence"] = max(
                    obj["confidence"], float(existence[slot_idx])
                )

        # Convert predictions to events
        clip_events = predictions_to_events(
            predictions,
            track_id_map=track_id_map,
            action_names=action_names,
            frame_indices=frame_indices,
            confidence_threshold=confidence_threshold,
        )
        all_events.extend(clip_events)

    # Deduplicate events from overlapping clips
    deduped_events = _deduplicate_events(all_events)
    logger.info(
        "Total events: %d (before dedup: %d)", len(deduped_events), len(all_events)
    )

    # Build EventGraph
    tracked_objects = list(all_object_info.values())
    graph = build_event_graph(
        video_id=video_path.stem,
        tracked_objects=tracked_objects,
        events=deduped_events,
        metadata={
            "checkpoint": str(checkpoint_path),
            "confidence_threshold": confidence_threshold,
            "num_frames": len(frames),
            "num_clips": max(
                0, (len(frames) - clip_length) // clip_stride + 1
            ),
        },
    )

    return graph.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V-JEPA event graph inference on a video"
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
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vjepa_training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--vjepa-config",
        type=str,
        default="configs/vjepa.yaml",
        help="Path to V-JEPA feature extraction config",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to experiment override YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/event_graph.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum interaction score to keep an event",
    )
    parser.add_argument(
        "--actions-config",
        type=str,
        default="configs/actions.yaml",
        help="Path to actions vocabulary YAML",
    )
    parser.add_argument(
        "--vocab-config",
        type=str,
        default="configs/vocab.yaml",
        help="Path to object category vocabulary YAML",
    )
    args = parser.parse_args()

    setup_logging()

    # Load and merge configs
    config = Config.from_yaml(args.config)
    # Merge vjepa config for encoder settings
    vjepa_only = Config.from_yaml(args.vjepa_config)
    config.vjepa = vjepa_only.vjepa
    if args.override:
        config = config.merge(args.override)

    # Run inference
    result = run_inference(
        video_path=Path(args.video),
        checkpoint_path=Path(args.checkpoint),
        config=config,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        actions_config=args.actions_config,
        vocab_config=args.vocab_config,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Saved EventGraph to %s", output_path)
    print(f"\nEventGraph: {len(result['nodes'])} objects, {len(result['edges'])} events")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
