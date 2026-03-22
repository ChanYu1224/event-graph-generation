"""Extract V-JEPA features from videos and save as .pt files.

Supports two backends:
  - "hub" (default): PyTorch Hub, V-JEPA 2.1 (384px, ViT-L or ViT-g)
  - "hf": HuggingFace transformers, V-JEPA 2.0 (256px, ViT-L)

Usage:
    uv run python scripts/0_extract_vjepa_features.py \
        --config configs/vjepa.yaml \
        --video-dir data/resized/room \
        --output-dir data/vjepa_features_v21_vitl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.data.frame_sampler import FrameSampler
from event_graph_generation.utils.logging import setup_logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoProcessor

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


_VJEPA_CORRECT_URL = "https://dl.fbaipublicfiles.com/vjepa2"


def _patch_vjepa_hub_url() -> None:
    """Patch V-JEPA 2 hub source to use the correct checkpoint URL.

    The upstream repo uses localhost:8300 for testing. This patches the
    cached hub source to use the public download URL instead.
    """
    hub_dir = Path(torch.hub.get_dir())
    backbones_path = hub_dir / "facebookresearch_vjepa2_main" / "src" / "hub" / "backbones.py"
    if not backbones_path.exists():
        return
    content = backbones_path.read_text()
    if "localhost:8300" in content:
        content = content.replace(
            'VJEPA_BASE_URL = "http://localhost:8300"',
            f'VJEPA_BASE_URL = "{_VJEPA_CORRECT_URL}"',
        )
        backbones_path.write_text(content)
        # Clear cached imports so the patched version is used
        for key in list(sys.modules.keys()):
            if "facebookresearch_vjepa2" in key or "src.hub" in key:
                del sys.modules[key]
        logger.info("Patched V-JEPA hub URL to use dl.fbaipublicfiles.com")


def _bgr_to_pil(bgr_array: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB Image."""
    rgb = bgr_array[:, :, ::-1]
    return Image.fromarray(rgb)


def extract_features_for_video(
    video_path: Path,
    output_dir: Path,
    model: torch.nn.Module,
    processor: object,
    clip_length: int = 16,
    clip_stride: int = 8,
    target_fps: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "hub",
) -> int:
    """Extract V-JEPA features for all clips in a video.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save per-clip .pt files.
        model: V-JEPA model (encoder).
        processor: V-JEPA processor for frame preprocessing.
        clip_length: Number of frames per clip.
        clip_stride: Stride between clips.
        target_fps: Frame sampling rate.
        device: Device to run inference on.
        dtype: Model dtype.
        backend: "hub" for PyTorch Hub (2.1) or "hf" for HuggingFace (2.0).

    Returns:
        Number of clips processed.
    """
    video_id = video_path.stem
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Sample frames
    sampler = FrameSampler(target_fps=target_fps)
    frames = sampler.sample(video_path)

    if len(frames) < clip_length:
        logger.warning(
            "Video %s has only %d frames (need %d), skipping",
            video_id, len(frames), clip_length,
        )
        return 0

    num_clips = 0
    for clip_start in range(0, len(frames) - clip_length + 1, clip_stride):
        clip_idx = clip_start // clip_stride
        output_path = video_output_dir / f"clip_{clip_idx:04d}.pt"

        if output_path.exists():
            num_clips += 1
            continue

        clip_frames = frames[clip_start : clip_start + clip_length]
        frame_indices = [f.frame_index for f in clip_frames]

        if backend == "hub":
            # Hub preprocessor expects list of PIL Images
            images = [_bgr_to_pil(f.image) for f in clip_frames]
            clip_tensor = processor(images)  # list[Tensor(C,T,H,W)]
            clip_tensor = clip_tensor[0].unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad(), torch.autocast(device, dtype=dtype):
                tokens = model(clip_tensor).squeeze(0).float().cpu()  # (S, D)
        else:  # "hf"
            # HF processor expects list of RGB numpy arrays
            images = [f.image[:, :, ::-1] for f in clip_frames]
            inputs = processor(videos=images, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(device=device, dtype=dtype)
            with torch.no_grad():
                outputs = model(pixel_values)
                if hasattr(outputs, "last_hidden_state"):
                    tokens = outputs.last_hidden_state.squeeze(0).cpu()  # (S, D)
                else:
                    tokens = outputs[0].squeeze(0).cpu()  # (S, D)

        torch.save(
            {
                "vjepa_tokens": tokens,
                "video_id": video_id,
                "clip_index": clip_idx,
                "frame_indices": frame_indices,
            },
            output_path,
        )
        num_clips += 1

    logger.info("Extracted %d clips from %s", num_clips, video_id)
    return num_clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V-JEPA features")
    parser.add_argument("--config", type=str, default="configs/vjepa.yaml")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    setup_logging()
    config = Config.from_yaml(args.config)

    video_dir = Path(args.video_dir or "data/resized/room")
    output_dir = Path(args.output_dir or config.vjepa.features_dir)
    backend = config.vjepa.backend

    device = args.device
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.vjepa.dtype, torch.bfloat16)

    if backend == "hub":
        logger.info(
            "Loading V-JEPA model via PyTorch Hub: %s/%s",
            config.vjepa.hub_repo, config.vjepa.hub_model_name,
        )
        # Ensure hub repo is cached, then patch the checkpoint URL
        torch.hub.list(config.vjepa.hub_repo, force_reload=False)
        _patch_vjepa_hub_url()

        processor = torch.hub.load(
            config.vjepa.hub_repo, "vjepa2_preprocessor",
            crop_size=config.vjepa.image_size,
        )
        encoder, _predictor = torch.hub.load(
            config.vjepa.hub_repo, config.vjepa.hub_model_name,
        )
        model = encoder.to(device=device, dtype=dtype).eval()
    else:  # "hf"
        if not _TRANSFORMERS_AVAILABLE:
            logger.error("transformers is required for V-JEPA HF backend")
            sys.exit(1)
        logger.info("Loading V-JEPA model via HuggingFace: %s", config.vjepa.model_name)
        processor = AutoProcessor.from_pretrained(config.vjepa.model_name)
        model = AutoModel.from_pretrained(
            config.vjepa.model_name, torch_dtype=dtype
        ).to(device).eval()

    all_videos = sorted(video_dir.glob("*.mp4"))
    video_files = [v for i, v in enumerate(all_videos) if i % args.num_shards == args.shard_id]
    logger.info(
        "Shard %d/%d: processing %d/%d videos from %s",
        args.shard_id, args.num_shards, len(video_files), len(all_videos), video_dir,
    )

    total_clips = 0
    for video_path in video_files:
        n = extract_features_for_video(
            video_path=video_path,
            output_dir=output_dir,
            model=model,
            processor=processor,
            clip_length=config.vjepa.frames_per_clip,
            clip_stride=config.vjepa.frames_per_clip // 2,
            target_fps=1.0,  # V-JEPA expects 1fps sampled frames
            device=device,
            dtype=dtype,
            backend=backend,
        )
        total_clips += n

    logger.info("Total clips extracted: %d", total_clips)


if __name__ == "__main__":
    main()
