"""Batch collation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Batch:
    """Container for a collated batch. Add fields as needed."""

    inputs: torch.Tensor
    targets: torch.Tensor


def collate_fn(samples: list[Any]) -> Batch:
    """Custom collate function for DataLoader.

    Adapt this to your data format.
    """
    raise NotImplementedError("Implement collate_fn for your data format.")
