"""Extract frames from selected clips for manual annotation.

Produces individual frame PNGs and a 4x4 contact sheet per clip.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a label overlay on the top-left of a frame.

    Args:
        frame: BGR image array.
        text: Label text to draw.

    Returns:
        Frame with label drawn.
    """
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (0, 0), (tw + 10, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (5, th + 7), font, scale, (255, 255, 255), thickness)
    return frame


def _make_contact_sheet(frames: list[np.ndarray], ncols: int = 4) -> np.ndarray:
    """Create a grid contact sheet from a list of frames.

    Args:
        frames: List of BGR images (all same size).
        ncols: Number of columns in the grid.

    Returns:
        Contact sheet image.
    """
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = frames[0].shape[:2]
    nrows = (len(frames) + ncols - 1) // ncols

    # Pad to fill grid
    while len(frames) < nrows * ncols:
        frames.append(np.zeros_like(frames[0]))

    rows = []
    for r in range(nrows):
        row_frames = frames[r * ncols:(r + 1) * ncols]
        rows.append(np.hstack(row_frames))
    return np.vstack(rows)


def extract_clip_frames(
    video_path: str,
    frame_indices: list[int],
    output_dir: Path,
) -> None:
    """Extract frames for a single clip and create a contact sheet.

    Args:
        video_path: Path to the source video file.
        frame_indices: List of source frame indices to extract.
        output_dir: Directory to save extracted frames and contact sheet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    labeled_frames = []
    for local_idx, source_frame in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                "Failed to read frame %d from %s", source_frame, video_path,
            )
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Save individual frame
        frame_path = output_dir / f"frame_{local_idx:02d}.png"
        cv2.imwrite(str(frame_path), frame)

        # Create labeled version for contact sheet
        label = f"L{local_idx:02d} / F{source_frame}"
        labeled = _draw_label(frame, label)
        labeled_frames.append(labeled)

    cap.release()

    # Create contact sheet
    if labeled_frames:
        sheet = _make_contact_sheet(labeled_frames, ncols=4)
        sheet_path = output_dir / "contact_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        logger.info("  Contact sheet: %s (%dx%d)", sheet_path, sheet.shape[1], sheet.shape[0])


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Extract frames for VLM quality eval")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/vlm_quality/selected_clips.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/vlm_quality/frames"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.manifest) as f:
        manifest = json.load(f)

    clips = manifest["clips"]
    logger.info("Extracting frames for %d clips", len(clips))

    for i, clip in enumerate(clips):
        video_id = clip["video_id"]
        clip_index = clip["clip_index"]
        frame_indices = clip["frame_indices"]
        video_path = clip["video_path"]

        clip_dir = args.output_dir / f"{video_id}_clip{clip_index:04d}"
        logger.info(
            "[%d/%d] %s clip %d (%d frames)",
            i + 1, len(clips), video_id, clip_index, len(frame_indices),
        )

        if not Path(video_path).exists():
            logger.error("  Video not found: %s", video_path)
            continue

        extract_clip_frames(video_path, frame_indices, clip_dir)

    logger.info("Done. Frames saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
