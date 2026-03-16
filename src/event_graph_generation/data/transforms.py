"""Data augmentation and preprocessing transforms."""

from __future__ import annotations

from typing import Any


def build_transforms(split: str = "train") -> Any:
    """Build data transforms for a given split.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        A callable transform or a composition of transforms.
    """
    raise NotImplementedError("Implement transforms for your data format.")
