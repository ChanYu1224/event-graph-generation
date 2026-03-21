"""Motion detection utilities for filtering static video clips."""

from __future__ import annotations

import cv2
import numpy as np


def compute_clip_motion_score(frames: list[np.ndarray]) -> float:
    """Compute motion score for a clip by comparing sampled frames.

    Samples the first, middle, and last frames, converts to grayscale,
    and computes the mean absolute difference between consecutive pairs.
    Returns the maximum of the two differences as the motion score.

    Args:
        frames: List of BGR numpy arrays (H, W, 3).

    Returns:
        Motion score in 0-255 scale. Higher values indicate more motion.
    """
    if len(frames) < 2:
        return 0.0

    mid = len(frames) // 2
    sampled = [frames[0], frames[mid], frames[-1]]

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sampled]

    scores = []
    for i in range(len(gray) - 1):
        diff = cv2.absdiff(gray[i], gray[i + 1])
        scores.append(float(np.mean(diff)))

    return max(scores)
