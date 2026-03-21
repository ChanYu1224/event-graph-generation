"""Video frame sampling using OpenCV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A single sampled frame with metadata."""

    image: np.ndarray  # (H, W, 3) BGR
    frame_index: int  # Original frame number in the video
    timestamp_sec: float  # Timestamp in seconds


class FrameSampler:
    """Extract frames from a video at a specified FPS."""

    def __init__(self, target_fps: float = 1.0) -> None:
        """
        Args:
            target_fps: Frames per second to extract. Default 1.0.
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        self.target_fps = target_fps

    def sample(self, video_path: str | Path) -> list[SampledFrame]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of SampledFrame objects.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            return self._extract_frames(cap)
        finally:
            cap.release()

    def _extract_frames(self, cap: cv2.VideoCapture) -> list[SampledFrame]:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            raise RuntimeError("Cannot determine video FPS")

        # Calculate frame interval and target indices
        frame_interval = max(1, round(source_fps / self.target_fps))
        target_indices = set(range(0, total_frames, frame_interval))
        frames: list[SampledFrame] = []

        # Sequential read: grab() skips non-target frames without full decode
        for frame_idx in range(total_frames):
            if frame_idx in target_indices:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(
                    SampledFrame(
                        image=frame,
                        frame_index=frame_idx,
                        timestamp_sec=frame_idx / source_fps,
                    )
                )
            else:
                if not cap.grab():
                    break

        logger.info(
            "Sampled %s frames from %s total (source_fps=%.1f, target_fps=%s)",
            len(frames), total_frames, source_fps, self.target_fps,
        )
        return frames

    def sample_clip(
        self,
        video_path: str | Path,
        start_frame: int,
        num_frames: int,
    ) -> list[SampledFrame]:
        """
        Extract a clip of frames starting at start_frame.

        Args:
            video_path: Path to the video file.
            start_frame: Frame index to start from.
            num_frames: Number of frames to extract at target_fps.

        Returns:
            List of SampledFrame objects.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0:
                raise RuntimeError("Cannot determine video FPS")

            frame_interval = max(1, round(source_fps / self.target_fps))
            frames: list[SampledFrame] = []

            # Build target frame indices
            target_indices = set(
                start_frame + i * frame_interval for i in range(num_frames)
            )
            end_frame = max(target_indices) + 1 if target_indices else start_frame

            # Seek once to start_frame, then read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame):
                if frame_idx in target_indices:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(
                        SampledFrame(
                            image=frame,
                            frame_index=frame_idx,
                            timestamp_sec=frame_idx / source_fps,
                        )
                    )
                else:
                    if not cap.grab():
                        break

            return frames
        finally:
            cap.release()

    @staticmethod
    def get_video_info(video_path: str | Path) -> dict:
        """Return basic video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            return {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6),
            }
        finally:
            cap.release()
