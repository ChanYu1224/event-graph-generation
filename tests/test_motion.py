"""Tests for motion detection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from event_graph_generation.utils.motion import compute_clip_motion_score


def _make_static_clip(n_frames: int = 16, height: int = 64, width: int = 64) -> list[np.ndarray]:
    """Create a clip of identical BGR frames."""
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    return [frame.copy() for _ in range(n_frames)]


def _make_motion_clip(
    n_frames: int = 16,
    height: int = 64,
    width: int = 64,
    delta: int = 50,
) -> list[np.ndarray]:
    """Create a clip with linearly increasing brightness."""
    frames = []
    for i in range(n_frames):
        val = int(50 + (delta * i / max(n_frames - 1, 1)))
        frame = np.full((height, width, 3), val, dtype=np.uint8)
        frames.append(frame)
    return frames


class TestComputeClipMotionScore:
    """Tests for compute_clip_motion_score."""

    def test_static_clip_returns_near_zero(self) -> None:
        clip = _make_static_clip()
        score = compute_clip_motion_score(clip)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_motion_clip_returns_high_score(self) -> None:
        clip = _make_motion_clip(delta=50)
        score = compute_clip_motion_score(clip)
        assert score > 10.0

    def test_empty_list_returns_zero(self) -> None:
        score = compute_clip_motion_score([])
        assert score == 0.0

    def test_single_frame_returns_zero(self) -> None:
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        score = compute_clip_motion_score([frame])
        assert score == 0.0

    def test_two_identical_frames_returns_zero(self) -> None:
        frame = np.full((64, 64, 3), 100, dtype=np.uint8)
        score = compute_clip_motion_score([frame.copy(), frame.copy()])
        assert score == pytest.approx(0.0, abs=0.1)

    def test_two_different_frames_returns_positive(self) -> None:
        frame_a = np.full((64, 64, 3), 50, dtype=np.uint8)
        frame_b = np.full((64, 64, 3), 200, dtype=np.uint8)
        score = compute_clip_motion_score([frame_a, frame_b])
        assert score > 5.0

    def test_first_half_motion_detected(self) -> None:
        """Motion only in first half should still yield a high score."""
        n = 16
        frames = []
        for i in range(n):
            if i < n // 2:
                val = 50 + i * 10
            else:
                val = 50 + (n // 2 - 1) * 10
            frames.append(np.full((64, 64, 3), val, dtype=np.uint8))
        score = compute_clip_motion_score(frames)
        assert score > 5.0
