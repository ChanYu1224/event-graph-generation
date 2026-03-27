"""Slot Attention analysis: visualize attention patterns from a trained ObjectPoolingModule.

Includes both standalone attention analysis and video-overlay visualizations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path for src layout
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.models.base import build_model
from event_graph_generation.utils.io import load_checkpoint

logger = logging.getLogger(__name__)

# 28 categories + unknown
CATEGORIES = [
    "person", "hand", "chair", "desk", "laptop", "monitor", "phone",
    "keyboard", "mouse", "tablet", "pen", "notebook", "book", "bookshelf",
    "shelf", "cup", "drawer", "curtain", "jacket", "backpack", "box",
    "speaker", "microphone", "stool", "pc_case", "earbuds", "smartphone",
    "case", "unknown",
]

# Spatial grid size (V-JEPA 2.1 ViT-L 384px: 24x24 patches)
H = W = 24
T = 8  # temporal tokens
K = 24  # num slots
FRAMES_PER_TEMPORAL_TOKEN = 2  # V-JEPA temporal downsampling: 16 frames -> 8 tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slot Attention analysis")
    parser.add_argument(
        "--config", default="configs/vjepa_training.yaml", help="Model config YAML"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vjepa_vitl_20260323_094236/best.pt",
        help="Trained checkpoint path",
    )
    parser.add_argument(
        "--features-dir",
        default="data/vjepa_features_v21_vitl",
        help="Directory with V-JEPA feature .pt files",
    )
    parser.add_argument(
        "--video-dir",
        default="data/resized/room",
        help="Directory with source video .mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        default="output/slot_attention_analysis",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8, help="Number of samples to analyze"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--overlay-clips",
        nargs="*",
        default=None,
        help="Specific clips for overlay, e.g. 20260317_095539_tp00033/clip_0130",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(
    features_dir: Path, num_samples: int
) -> tuple[list[torch.Tensor], list[str], list[dict]]:
    """Load one clip per video directory for diversity.

    Returns:
        tokens_list: V-JEPA tokens per sample.
        sample_ids: Human-readable sample identifiers.
        metadata_list: Per-sample metadata (video_id, clip_index, frame_indices).
    """
    video_dirs = sorted(d for d in features_dir.iterdir() if d.is_dir())
    tokens_list: list[torch.Tensor] = []
    sample_ids: list[str] = []
    metadata_list: list[dict] = []

    for vdir in video_dirs[:num_samples]:
        clips = sorted(vdir.glob("clip_*.pt"))
        if not clips:
            continue
        # Pick a clip from the middle of the video
        clip_path = clips[len(clips) // 2]
        data = torch.load(clip_path, map_location="cpu", weights_only=True)
        tokens_list.append(data["vjepa_tokens"])  # (S, D)
        sample_ids.append(f"{vdir.name}/{clip_path.stem}")
        metadata_list.append({
            "video_id": data.get("video_id", vdir.name),
            "clip_index": data.get("clip_index", 0),
            "frame_indices": data.get("frame_indices", []),
        })

    logger.info("Loaded %d samples", len(tokens_list))
    return tokens_list, sample_ids, metadata_list


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def run_forward(
    model: torch.nn.Module,
    tokens_list: list[torch.Tensor],
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Run ObjectPoolingModule on each sample, collect results."""
    object_pooling = model.object_pooling
    object_pooling.eval()

    all_attn: list[torch.Tensor] = []
    all_existence: list[torch.Tensor] = []
    all_categories: list[torch.Tensor] = []

    with torch.no_grad():
        for tokens in tokens_list:
            x = tokens.unsqueeze(0).to(device)  # (1, S, D)
            obj_repr = object_pooling(x)
            all_attn.append(obj_repr.attn_maps[0].cpu())  # (K, S)
            all_existence.append(obj_repr.existence[0].cpu())  # (K,)
            all_categories.append(obj_repr.categories[0].cpu())  # (K, n_cat)

    return all_attn, all_existence, all_categories


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _slot_colors(k: int = K) -> np.ndarray:
    """Generate k distinct colors for slots."""
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    extra = plt.cm.Set3(np.linspace(0, 1, max(k - 20, 4)))
    colors = np.vstack([cmap, extra])[:k]
    return colors


def _spatial_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Compute per-slot spatial entropy. attn: (K, S)."""
    attn_3d = attn.reshape(K, T, H * W)
    spatial = attn_3d.sum(dim=1)  # (K, H*W)
    p = spatial / (spatial.sum(dim=1, keepdim=True) + 1e-10)
    entropy = -(p * torch.log(p + 1e-10)).sum(dim=1)  # (K,)
    return entropy


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_spatial_heatmaps(
    attn: torch.Tensor, sample_id: str, output_dir: Path
) -> None:
    """4x6 grid of per-slot spatial attention heatmaps."""
    attn_3d = attn.reshape(K, T, H, W)
    spatial = attn_3d.sum(dim=1).numpy()  # (K, H, W) sum over time

    fig, axes = plt.subplots(4, 6, figsize=(20, 14))
    fig.suptitle(f"Spatial Attention Heatmaps\n{sample_id}", fontsize=14)

    for k in range(K):
        ax = axes[k // 6, k % 6]
        im = ax.imshow(spatial[k], cmap="viridis", interpolation="nearest")
        ax.set_title(f"Slot {k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_dir / "01_spatial_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 01_spatial_heatmaps.png")


def plot_slot_ownership(
    attn: torch.Tensor, sample_id: str, output_dir: Path
) -> None:
    """Slot ownership maps: which slot wins each spatial position."""
    attn_3d = attn.reshape(K, T, H, W)
    colors = _slot_colors()

    # Per-timestep ownership
    ownership_t = attn_3d.argmax(dim=0).numpy()  # (T, H, W)

    # Summary: mode across time
    ownership_mode = torch.mode(attn_3d.argmax(dim=0), dim=0).values.numpy()  # (H, W)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"Slot Ownership Maps\n{sample_id}", fontsize=14)

    # First row: 8 timesteps + 1 empty + 1 summary
    for t in range(T):
        ax = axes[t // 5, t % 5]
        rgb = colors[ownership_t[t]].reshape(H, W, -1)[:, :, :3]
        ax.imshow(rgb)
        ax.set_title(f"t={t}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Summary in last position
    ax_sum = axes[1, 4]
    rgb_sum = colors[ownership_mode].reshape(H, W, -1)[:, :, :3]
    ax_sum.imshow(rgb_sum)
    ax_sum.set_title("Mode (summary)", fontsize=10)
    ax_sum.set_xticks([])
    ax_sum.set_yticks([])

    # Hide unused subplot
    axes[1, 3].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "02_slot_ownership.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 02_slot_ownership.png")


def plot_temporal_profiles(
    all_attn: list[torch.Tensor], sample_ids: list[str], output_dir: Path
) -> None:
    """Temporal attention profiles: per-slot attention mass over timesteps."""
    colors = _slot_colors()

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Temporal Attention Profiles", fontsize=14)

    # --- Single sample (first) ---
    attn = all_attn[0]
    attn_3d = attn.reshape(K, T, H * W)
    temporal = attn_3d.sum(dim=2).numpy()  # (K, T)

    ax = axes[0]
    for k in range(K):
        ax.plot(range(T), temporal[k], color=colors[k], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total attention mass")
    ax.set_title(f"Single sample: {sample_ids[0]}", fontsize=10)
    ax.set_xticks(range(T))

    # --- Stacked area (first sample) ---
    ax = axes[1]
    ax.stackplot(range(T), temporal, colors=colors[:K], alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative attention mass")
    ax.set_title("Stacked area (single sample)", fontsize=10)
    ax.set_xticks(range(T))

    # --- Mean across all samples ---
    all_temporal = []
    for a in all_attn:
        a3d = a.reshape(K, T, H * W)
        all_temporal.append(a3d.sum(dim=2).numpy())
    mean_temporal = np.mean(all_temporal, axis=0)  # (K, T)
    std_temporal = np.std(all_temporal, axis=0)

    ax = axes[2]
    for k in range(K):
        ax.plot(range(T), mean_temporal[k], color=colors[k], alpha=0.7, linewidth=1.5)
        ax.fill_between(
            range(T),
            mean_temporal[k] - std_temporal[k],
            mean_temporal[k] + std_temporal[k],
            color=colors[k],
            alpha=0.15,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total attention mass")
    ax.set_title(f"Mean +/- std ({len(all_attn)} samples)", fontsize=10)
    ax.set_xticks(range(T))

    plt.tight_layout()
    fig.savefig(output_dir / "03_temporal_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 03_temporal_profiles.png")


def plot_slot_entropy(
    all_attn: list[torch.Tensor],
    all_existence: list[torch.Tensor],
    output_dir: Path,
) -> None:
    """Per-slot spatial entropy bar chart with existence coloring."""
    entropies = torch.stack([_spatial_entropy(a) for a in all_attn])  # (N, K)
    mean_ent = entropies.mean(dim=0).numpy()  # (K,)
    std_ent = entropies.std(dim=0).numpy()

    mean_exist = torch.stack(all_existence).mean(dim=0).numpy()  # (K,)
    max_entropy = float(np.log(H * W))

    # Sort by mean entropy
    order = np.argsort(mean_ent)

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = plt.cm.RdYlGn(mean_exist[order])  # green=high existence

    ax.bar(range(K), mean_ent[order], yerr=std_ent[order], color=bar_colors,
           edgecolor="gray", capsize=3)
    ax.axhline(max_entropy, color="red", linestyle="--", alpha=0.6,
               label=f"Max entropy (uniform) = {max_entropy:.2f}")
    ax.axhline(max_entropy / 2, color="orange", linestyle=":", alpha=0.6,
               label=f"Half max = {max_entropy / 2:.2f}")
    ax.set_xlabel("Slot (sorted by entropy)")
    ax.set_ylabel("Spatial entropy")
    ax.set_title(f"Slot Spatial Entropy (mean +/- std, {len(all_attn)} samples)\n"
                 "Color: green=high existence, red=low existence")
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(s) for s in order], fontsize=8)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "04_slot_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 04_slot_entropy.png")


def plot_slot_competition(
    attn: torch.Tensor, sample_id: str, output_dir: Path
) -> None:
    """Show attention distribution across slots for selected tokens."""
    colors = _slot_colors()

    # Per-token entropy across slots (attn already softmaxed on slot axis)
    token_entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=0)  # (S,)

    # Select 5 lowest (one slot dominates) + 5 highest (distributed)
    n_select = 5
    low_idx = token_entropy.topk(n_select, largest=False).indices
    high_idx = token_entropy.topk(n_select, largest=True).indices
    selected = torch.cat([low_idx, high_idx])

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(f"Slot Competition at Selected Token Positions\n{sample_id}",
                 fontsize=13)

    x = np.arange(len(selected))
    bottom = np.zeros(len(selected))

    for k in range(K):
        vals = attn[k, selected].numpy()
        ax.bar(x, vals, bottom=bottom, color=colors[k], width=0.8,
               label=f"Slot {k}" if k < 6 else None)
        bottom += vals

    # X-axis labels: (t, y, x) coordinates
    labels = []
    for idx in selected:
        i = idx.item()
        t_pos = i // (H * W)
        spatial_pos = i % (H * W)
        y_pos = spatial_pos // W
        x_pos = spatial_pos % W
        labels.append(f"({t_pos},{y_pos},{x_pos})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_xlabel("Token position (t, y, x)")
    ax.set_ylabel("Attention weight (sums to 1)")

    # Add vertical separator
    ax.axvline(n_select - 0.5, color="black", linestyle="--", alpha=0.5)
    ax.text(n_select / 2 - 0.5, 1.02, "Low entropy\n(1 slot dominates)",
            ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(n_select + n_select / 2 - 0.5, 1.02, "High entropy\n(distributed)",
            ha="center", fontsize=9, transform=ax.get_xaxis_transform())

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "05_slot_competition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 05_slot_competition.png")


def plot_existence_and_categories(
    all_existence: list[torch.Tensor],
    all_categories: list[torch.Tensor],
    all_attn: list[torch.Tensor],
    sample_ids: list[str],
    output_dir: Path,
) -> None:
    """Existence scores, existence vs entropy, and category predictions."""
    n_samples = len(all_existence)
    exist_mat = torch.stack(all_existence).numpy()  # (N, K)
    mean_exist = exist_mat.mean(axis=0)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # --- (a) Existence scores per slot, all samples ---
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(K)
    width = 0.8 / n_samples
    for i in range(n_samples):
        offset = (i - n_samples / 2 + 0.5) * width
        ax.bar(x + offset, exist_mat[i], width, alpha=0.7,
               label=sample_ids[i].split("/")[0][:12])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Threshold 0.5")
    ax.set_xlabel("Slot")
    ax.set_ylabel("Existence score")
    ax.set_title("(a) Existence Scores per Slot")
    ax.set_xticks(x)
    ax.legend(fontsize=6, ncol=2)

    # --- (b) Active slot count histogram ---
    ax = fig.add_subplot(gs[0, 1])
    active_counts = (exist_mat > 0.5).sum(axis=1)
    ax.bar(range(n_samples), active_counts, color="steelblue")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Active slots (existence > 0.5)")
    ax.set_title(f"(b) Active Slots per Sample (mean={active_counts.mean():.1f})")
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels([s.split("/")[0][:10] for s in sample_ids],
                       fontsize=7, rotation=45)

    # --- (c) Existence vs Entropy scatter ---
    ax = fig.add_subplot(gs[1, 0])
    entropies = torch.stack([_spatial_entropy(a) for a in all_attn]).numpy()  # (N, K)
    for i in range(n_samples):
        ax.scatter(entropies[i], exist_mat[i], alpha=0.5, s=20,
                   label=sample_ids[i].split("/")[0][:12] if i < 4 else None)
    ax.set_xlabel("Spatial entropy")
    ax.set_ylabel("Existence score")
    ax.set_title("(c) Existence vs Spatial Entropy")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    ax.legend(fontsize=7)

    # --- (d) Category distribution for active slots ---
    ax = fig.add_subplot(gs[1, 1])
    cat_counts = np.zeros(len(CATEGORIES))
    for i in range(n_samples):
        active_mask = exist_mat[i] > 0.5
        if active_mask.any():
            cat_logits = all_categories[i][active_mask]  # (n_active, n_cat)
            cat_preds = cat_logits.argmax(dim=1).numpy()
            for c in cat_preds:
                cat_counts[c] += 1

    nonzero = cat_counts > 0
    ax.barh(np.arange(len(CATEGORIES))[nonzero], cat_counts[nonzero],
            color="steelblue")
    ax.set_yticks(np.arange(len(CATEGORIES))[nonzero])
    ax.set_yticklabels([CATEGORIES[i] for i in range(len(CATEGORIES)) if nonzero[i]],
                       fontsize=9)
    ax.set_xlabel("Count (across all active slots and samples)")
    ax.set_title("(d) Predicted Categories for Active Slots")

    fig.suptitle("Existence Scores & Category Analysis", fontsize=15, y=1.01)
    fig.savefig(output_dir / "06_existence_and_categories.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 06_existence_and_categories.png")


def plot_cross_sample_consistency(
    all_attn: list[torch.Tensor], output_dir: Path
) -> None:
    """Cross-sample consistency: cosine similarity of spatial attention per slot."""
    n_samples = len(all_attn)

    # For each slot, compute spatial attention profile (K, H*W) per sample
    spatial_profiles = []
    for attn in all_attn:
        attn_3d = attn.reshape(K, T, H * W)
        spatial = attn_3d.sum(dim=1)  # (K, H*W)
        spatial = F.normalize(spatial, dim=1)  # L2-normalize
        spatial_profiles.append(spatial)

    # Per-slot consistency: mean pairwise cosine similarity across samples
    consistency = np.zeros(K)
    pair_count = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # (K,) cosine similarity per slot
            cos_sim = (spatial_profiles[i] * spatial_profiles[j]).sum(dim=1).numpy()
            consistency += cos_sim
            pair_count += 1
    consistency /= max(pair_count, 1)

    # Slot-slot similarity matrix (averaged across samples)
    all_spatial = torch.stack(spatial_profiles)  # (N, K, H*W)
    mean_spatial = all_spatial.mean(dim=0)  # (K, H*W)
    sim_matrix = torch.mm(mean_spatial, mean_spatial.t()).numpy()  # (K, K)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- (a) Per-slot consistency bar ---
    ax = axes[0]
    order = np.argsort(consistency)[::-1]
    ax.bar(range(K), consistency[order], color="steelblue", edgecolor="gray")
    ax.set_xlabel("Slot (sorted by consistency)")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("(a) Cross-Sample Spatial Consistency per Slot")
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(s) for s in order], fontsize=8)
    ax.set_ylim(0, 1)

    # --- (b) Slot-slot similarity matrix ---
    ax = axes[1]
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Slot")
    ax.set_title("(b) Slot-Slot Cosine Similarity (mean spatial profile)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Cross-Sample Consistency ({len(all_attn)} samples)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "07_cross_sample_consistency.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 07_cross_sample_consistency.png")


# ---------------------------------------------------------------------------
# Video frame helpers
# ---------------------------------------------------------------------------

def extract_frames(
    video_dir: Path, video_id: str, frame_indices: list[int]
) -> list[np.ndarray] | None:
    """Extract specific frames from a video file.

    Returns:
        List of BGR frames as numpy arrays, or None if the video is missing.
    """
    video_path = video_dir / f"{video_id}.mp4"
    if not video_path.exists():
        logger.warning("Video not found: %s", video_path)
        return None

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((568, 1008, 3), dtype=np.uint8))
    cap.release()
    return frames


def _temporal_token_frames(frame_indices: list[int]) -> list[int]:
    """Map temporal tokens to representative frame indices.

    V-JEPA uses 16 frames -> 8 temporal tokens (2 frames per token).
    Returns the frame index for the first frame of each token's window.
    """
    indices = []
    for t in range(T):
        fi = t * FRAMES_PER_TEMPORAL_TOKEN
        if fi < len(frame_indices):
            indices.append(frame_indices[fi])
        elif frame_indices:
            indices.append(frame_indices[-1])
    return indices


def _overlay_heatmap(
    frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay a heatmap on a video frame.

    Args:
        frame: BGR image (H_frame, W_frame, 3).
        heatmap: 2D array (H_grid, W_grid), values in [0, 1].
        alpha: Blending factor for the heatmap.

    Returns:
        Blended BGR image.
    """
    h, w = frame.shape[:2]
    # Resize heatmap to frame resolution
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 255]
    heatmap_u8 = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)


def _overlay_segmentation(
    frame: np.ndarray, ownership: np.ndarray, colors: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay slot ownership segmentation on a video frame."""
    h, w = frame.shape[:2]
    # ownership: (H_grid, W_grid) with slot indices
    # Resize using nearest-neighbor to preserve slot assignments
    ownership_resized = cv2.resize(
        ownership.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(int)
    # Build color image
    seg_rgb = (colors[ownership_resized][:, :, :3] * 255).astype(np.uint8)
    seg_bgr = seg_rgb[:, :, ::-1]
    return cv2.addWeighted(frame, 1 - alpha, seg_bgr, alpha, 0)


# ---------------------------------------------------------------------------
# Video overlay plot functions
# ---------------------------------------------------------------------------

def plot_overlay_heatmaps(
    attn: torch.Tensor,
    existence: torch.Tensor,
    categories: torch.Tensor,
    frames: list[np.ndarray],
    frame_indices: list[int],
    sample_id: str,
    output_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Overlay per-slot attention heatmaps on video frames (active slots only).

    Shows one representative frame (middle timestep) with each active slot's
    attention heatmap overlaid.
    """
    attn_3d = attn.reshape(K, T, H, W)  # (K, T, 24, 24)

    # Pick active slots (existence > 0.5), sorted by existence descending
    active_mask = existence > 0.5
    active_indices = torch.where(active_mask)[0]
    active_indices = active_indices[existence[active_indices].argsort(descending=True)]
    n_active = len(active_indices)

    if n_active == 0:
        logger.warning("No active slots for %s, skipping overlay heatmaps", sample_id)
        return

    # Use middle timestep (t=3 or 4) as representative frame
    t_mid = T // 2
    frame_idx_in_clip = t_mid * FRAMES_PER_TEMPORAL_TOKEN
    if frame_idx_in_clip < len(frames):
        ref_frame = frames[frame_idx_in_clip]
    else:
        ref_frame = frames[len(frames) // 2]

    # Grid layout
    n_cols = min(6, n_active)
    n_rows = (n_active + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Slot Attention Overlay (active slots, t={t_mid})\n{sample_id}",
        fontsize=13,
    )

    for i, slot_k in enumerate(active_indices):
        k = slot_k.item()
        ax = axes[i // n_cols, i % n_cols]

        # Get spatial heatmap for this slot at the middle timestep
        hm = attn_3d[k, t_mid].numpy()
        hm_norm = hm / (hm.max() + 1e-8)  # normalize to [0, 1]

        overlay = _overlay_heatmap(ref_frame, hm_norm, alpha=0.5)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax.imshow(overlay_rgb)

        cat_idx = categories[k].argmax().item()
        cat_name = CATEGORIES[cat_idx]
        ex_score = existence[k].item()
        ax.set_title(f"Slot {k}: {cat_name}\nexist={ex_score:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_active, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout()
    fname = f"08_overlay_heatmaps{suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_overlay_temporal(
    attn: torch.Tensor,
    existence: torch.Tensor,
    categories: torch.Tensor,
    frames: list[np.ndarray],
    frame_indices: list[int],
    sample_id: str,
    output_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Show top-4 active slots' attention across all 8 timesteps overlaid on frames."""
    attn_3d = attn.reshape(K, T, H, W)

    # Top-4 active slots by existence
    top_k = min(4, K)
    top_slots = existence.topk(top_k).indices

    fig, axes = plt.subplots(top_k, T, figsize=(2.5 * T, 3 * top_k))
    if top_k == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Slot Attention Temporal Evolution (top-{top_k} slots)\n{sample_id}",
        fontsize=13,
    )

    for row, slot_k in enumerate(top_slots):
        k = slot_k.item()
        cat_idx = categories[k].argmax().item()
        cat_name = CATEGORIES[cat_idx]

        for t in range(T):
            ax = axes[row, t]
            # Get the frame for this temporal token
            fi = t * FRAMES_PER_TEMPORAL_TOKEN
            frame = frames[fi] if fi < len(frames) else frames[-1]

            hm = attn_3d[k, t].numpy()
            hm_norm = hm / (hm.max() + 1e-8)

            overlay = _overlay_heatmap(frame, hm_norm, alpha=0.5)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])

            if t == 0:
                ax.set_ylabel(f"Slot {k}\n{cat_name}", fontsize=9)
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)

    plt.tight_layout()
    fname = f"09_overlay_temporal{suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_overlay_segmentation(
    attn: torch.Tensor,
    frames: list[np.ndarray],
    frame_indices: list[int],
    sample_id: str,
    output_dir: Path,
    *,
    suffix: str = "",
) -> None:
    """Overlay slot ownership segmentation on video frames across timesteps."""
    attn_3d = attn.reshape(K, T, H, W)
    colors = _slot_colors()

    # Ownership per timestep
    ownership_t = attn_3d.argmax(dim=0).numpy()  # (T, H, W)

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 7))
    fig.suptitle(
        f"Slot Ownership Overlay on Video Frames\n{sample_id}", fontsize=13
    )

    for t in range(T):
        fi = t * FRAMES_PER_TEMPORAL_TOKEN
        frame = frames[fi] if fi < len(frames) else frames[-1]

        # Top row: original frame
        ax = axes[0, t]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={t} (frame {frame_indices[fi] if fi < len(frame_indices) else '?'})",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel("Original", fontsize=10)

        # Bottom row: segmentation overlay
        ax = axes[1, t]
        overlay = _overlay_segmentation(frame, ownership_t[t], colors, alpha=0.45)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        if t == 0:
            ax.set_ylabel("Slot ownership", fontsize=10)

    plt.tight_layout()
    fname = f"10_overlay_segmentation{suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and build model
    logger.info("Loading config from %s", args.config)
    config = Config.from_yaml(args.config)

    logger.info("Building model")
    model = build_model(config.model, vjepa_config=config.vjepa)
    load_checkpoint(args.checkpoint, model, device=args.device)
    model.to(args.device)
    model.eval()
    logger.info("Model loaded from %s (device=%s)", args.checkpoint, args.device)

    # Load samples
    tokens_list, sample_ids, metadata_list = load_samples(
        Path(args.features_dir), args.num_samples
    )
    logger.info("Sample IDs: %s", sample_ids)

    # Forward pass
    logger.info("Running forward pass on %d samples...", len(tokens_list))
    all_attn, all_existence, all_categories = run_forward(model, tokens_list, args.device)

    # Generate standalone plots
    logger.info("Generating standalone visualizations...")
    plot_spatial_heatmaps(all_attn[0], sample_ids[0], output_dir)
    plot_slot_ownership(all_attn[0], sample_ids[0], output_dir)
    plot_temporal_profiles(all_attn, sample_ids, output_dir)
    plot_slot_entropy(all_attn, all_existence, output_dir)
    plot_slot_competition(all_attn[0], sample_ids[0], output_dir)
    plot_existence_and_categories(
        all_existence, all_categories, all_attn, sample_ids, output_dir
    )
    plot_cross_sample_consistency(all_attn, output_dir)

    # Generate video overlay plots
    video_dir = Path(args.video_dir)
    logger.info("Generating video overlay visualizations...")

    if args.overlay_clips:
        # Load and process explicitly specified clips
        overlay_tokens = []
        overlay_ids = []
        overlay_meta = []
        for clip_spec in args.overlay_clips:
            # e.g. "20260317_095539_tp00033/clip_0130"
            parts = clip_spec.split("/")
            vdir_name = parts[0]
            clip_name = parts[1] if len(parts) > 1 else "clip_0000"
            clip_path = Path(args.features_dir) / vdir_name / f"{clip_name}.pt"
            if not clip_path.exists():
                logger.warning("Clip not found: %s", clip_path)
                continue
            data = torch.load(clip_path, map_location="cpu", weights_only=True)
            overlay_tokens.append(data["vjepa_tokens"])
            overlay_ids.append(f"{vdir_name}/{clip_name}")
            overlay_meta.append({
                "video_id": data.get("video_id", vdir_name),
                "clip_index": data.get("clip_index", 0),
                "frame_indices": data.get("frame_indices", []),
            })

        if overlay_tokens:
            logger.info("Running forward pass on %d overlay clips...", len(overlay_tokens))
            ov_attn, ov_exist, ov_cats = run_forward(model, overlay_tokens, args.device)
        else:
            ov_attn, ov_exist, ov_cats, overlay_meta, overlay_ids = [], [], [], [], []
    else:
        # Use the samples already loaded (first one with available video)
        ov_attn, ov_exist, ov_cats = all_attn, all_existence, all_categories
        overlay_meta, overlay_ids = metadata_list, sample_ids

    overlay_count = 0
    for i, meta in enumerate(overlay_meta):
        frames = extract_frames(
            video_dir, meta["video_id"], meta["frame_indices"]
        )
        if frames is None:
            continue

        suffix = f"_{overlay_count}" if len(overlay_meta) > 1 else ""
        logger.info("Overlay %d: %s", overlay_count, overlay_ids[i])
        plot_overlay_heatmaps(
            ov_attn[i], ov_exist[i], ov_cats[i],
            frames, meta["frame_indices"], overlay_ids[i], output_dir,
            suffix=suffix,
        )
        plot_overlay_temporal(
            ov_attn[i], ov_exist[i], ov_cats[i],
            frames, meta["frame_indices"], overlay_ids[i], output_dir,
            suffix=suffix,
        )
        plot_overlay_segmentation(
            ov_attn[i], frames, meta["frame_indices"],
            overlay_ids[i], output_dir,
            suffix=suffix,
        )
        overlay_count += 1

    if overlay_count == 0:
        logger.warning(
            "No matching video found in %s; overlay plots skipped", video_dir
        )
    else:
        logger.info("Generated overlays for %d clips", overlay_count)

    # Print summary stats
    exist_mat = torch.stack(all_existence)
    active = (exist_mat > 0.5).float()
    entropies = torch.stack([_spatial_entropy(a) for a in all_attn])

    print("\n" + "=" * 60)
    print("SLOT ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Samples analyzed:       {len(sample_ids)}")
    print(f"Active slots (mean):    {active.sum(dim=1).mean():.1f} / {K}")
    print(f"Active slots (range):   {active.sum(dim=1).min():.0f} - {active.sum(dim=1).max():.0f}")
    print(f"Spatial entropy (mean): {entropies.mean():.3f}")
    print(f"Spatial entropy (max possible): {np.log(H * W):.3f}")
    print(f"Output directory:       {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
