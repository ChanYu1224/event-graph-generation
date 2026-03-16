"""PyTorch Dataset base implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset class.

    Override ``__len__`` and ``__getitem__`` for your task.
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples: list[Any] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load sample paths/metadata. Override for your data format."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError
