# Structure Reference

This document defines all template file contents for the scaffold-training skill.
`<pkg>` is a placeholder for the package name (project name with hyphens replaced by underscores).

---

## configs/default.yaml

```yaml
# Default experiment configuration
# Override per-experiment in configs/experiment/

data:
  raw_dir: data/raw
  processed_dir: data/processed
  splits_dir: data/splits
  batch_size: 32
  num_workers: 4
  pin_memory: true

model:
  name: base_model
  # Add model-specific parameters here
  # e.g., hidden_dim: 256

training:
  epochs: 100
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  optimizer: adam  # adam | adamw | sgd
  scheduler: cosine  # cosine | step | none
  scheduler_params:
    T_max: 100
  grad_clip_norm: 1.0
  seed: 42
  device: cuda
  checkpoint_dir: checkpoints
  save_every_n_epochs: 10
  early_stopping_patience: 0  # 0 = disabled

evaluation:
  metrics:
    - accuracy
  eval_every_n_epochs: 1

wandb:
  enabled: true
  project: "<pkg>"
  entity: null  # your wandb team/user
  tags: []
  notes: ""
```

---

## configs/experiment/.gitkeep

Empty file.

---

## src/\<pkg\>/\_\_init\_\_.py

```python
"""<pkg> - Deep learning training pipeline."""
```

---

## src/\<pkg\>/config.py

```python
"""Dataclass-based configuration management with YAML loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    name: str = "base_model"


@dataclass
class SchedulerParams:
    T_max: int = 100


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    scheduler_params: SchedulerParams = field(default_factory=SchedulerParams)
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 0


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    eval_every_n_epochs: int = 1


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = ""
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Config:
        cfg = cls()
        if "data" in d:
            cfg.data = _update_dataclass(cfg.data, d["data"])
        if "model" in d:
            cfg.model = _update_dataclass(cfg.model, d["model"])
        if "training" in d:
            training_d = d["training"]
            if "scheduler_params" in training_d:
                sp = _update_dataclass(
                    cfg.training.scheduler_params, training_d.pop("scheduler_params")
                )
                cfg.training = _update_dataclass(cfg.training, training_d)
                cfg.training.scheduler_params = sp
            else:
                cfg.training = _update_dataclass(cfg.training, training_d)
        if "evaluation" in d:
            cfg.evaluation = _update_dataclass(cfg.evaluation, d["evaluation"])
        if "wandb" in d:
            cfg.wandb = _update_dataclass(cfg.wandb, d["wandb"])
        return cfg

    def merge(self, override_path: str | Path) -> Config:
        """Return a new Config merged with overrides from another YAML file."""
        with open(override_path) as f:
            overrides = yaml.safe_load(f) or {}
        base = self.to_dict()
        merged = _deep_merge(base, overrides)
        return self._from_dict(merged)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to a plain dict."""
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _update_dataclass[T](instance: T, updates: dict[str, Any]) -> T:
    """Update a dataclass instance with values from a dict."""
    updated = copy.deepcopy(instance)
    for k, v in updates.items():
        if hasattr(updated, k):
            setattr(updated, k, v)
    return updated


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
```

---

## src/\<pkg\>/data/\_\_init\_\_.py

```python
"""Data loading and preprocessing."""
```

---

## src/\<pkg\>/data/dataset.py

```python
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
```

---

## src/\<pkg\>/data/collator.py

```python
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
```

---

## src/\<pkg\>/data/transforms.py

```python
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
```

---

## src/\<pkg\>/models/\_\_init\_\_.py

```python
"""Model definitions."""
```

---

## src/\<pkg\>/models/base.py

```python
"""Base model definition."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class BaseModel(nn.Module):
    """Base model skeleton. Replace with your architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Define layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(config: ModelConfig) -> nn.Module:
    """Factory function to build a model from config."""
    return BaseModel(config)
```

---

## src/\<pkg\>/training/\_\_init\_\_.py

```python
"""Training loop and optimization."""
```

---

## src/\<pkg\>/training/trainer.py

```python
"""Training loop."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..evaluation.evaluator import Evaluator
from ..utils.io import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Custom training loop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: Config,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.evaluator = evaluator

        self.optimizer = build_optimizer(model, config.training)
        self.scheduler = build_scheduler(self.optimizer, config.training)
        self.criterion = nn.CrossEntropyLoss()  # Replace as needed

        self.current_epoch = 0
        self.best_metric = float("-inf")

    def train(self) -> None:
        """Run the full training loop."""
        try:
            import wandb

            if self.config.wandb.enabled:
                wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    config=self.config.to_dict(),
                    tags=self.config.wandb.tags,
                    notes=self.config.wandb.notes,
                )
        except ImportError:
            pass

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            train_loss = self._train_one_epoch()
            logger.info(f"Epoch {epoch}/{self.config.training.epochs} - loss: {train_loss:.4f}")

            try:
                import wandb

                if self.config.wandb.enabled:
                    wandb.log({"train/loss": train_loss, "epoch": epoch})
            except ImportError:
                pass

            # Evaluation
            if (
                self.val_loader is not None
                and self.evaluator is not None
                and (epoch + 1) % self.config.evaluation.eval_every_n_epochs == 0
            ):
                metrics = self.evaluator.evaluate(self.model, self.val_loader)
                logger.info(f"Epoch {epoch} - val metrics: {metrics}")

                try:
                    import wandb

                    if self.config.wandb.enabled:
                        wandb.log({f"val/{k}": v for k, v in metrics.items()})
                except ImportError:
                    pass

            # Checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                ckpt_dir = Path(self.config.training.checkpoint_dir)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

        try:
            import wandb

            if self.config.wandb.enabled:
                wandb.finish()
        except ImportError:
            pass

    def _train_one_epoch(self) -> float:
        """Train for a single epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Adapt this to your batch format
            outputs = self.model(batch.inputs.to(self.config.training.device))
            loss = self.criterion(
                outputs, batch.targets.to(self.config.training.device)
            )

            loss.backward()

            if self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(num_batches, 1)

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            device=self.config.training.device,
        )
        self.current_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
```

---

## src/\<pkg\>/training/optimizer.py

```python
"""Optimizer and scheduler factory functions."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from ..config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Build optimizer from config."""
    name = config.optimizer.lower()
    params = model.parameters()

    if name == "adam":
        return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif name == "sgd":
        return optim.SGD(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: optim.Optimizer, config: TrainingConfig
) -> LRScheduler | None:
    """Build learning rate scheduler from config."""
    name = config.scheduler.lower()

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.scheduler_params.T_max)
    elif name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
```

---

## src/\<pkg\>/evaluation/\_\_init\_\_.py

```python
"""Evaluation metrics and loop."""
```

---

## src/\<pkg\>/evaluation/metrics.py

```python
"""Metric computation functions."""

from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy."""
    pred_labels = predictions.argmax(dim=-1)
    return (pred_labels == targets).float().mean().item()


METRIC_REGISTRY: dict[str, callable] = {
    "accuracy": accuracy,
}


def get_metric(name: str) -> callable:
    """Get a metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
```

---

## src/\<pkg\>/evaluation/evaluator.py

```python
"""Evaluation loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import EvaluationConfig
from .metrics import get_metric


class Evaluator:
    """Runs evaluation on a dataset."""

    def __init__(self, config: EvaluationConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.metrics = {name: get_metric(name) for name in config.metrics}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on the given dataloader."""
        model.eval()
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            outputs = model(batch.inputs.to(self.device))
            all_predictions.append(outputs.cpu())
            all_targets.append(batch.targets)

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(predictions, targets)

        return results
```

---

## src/\<pkg\>/utils/\_\_init\_\_.py

```python
"""Utility functions."""
```

---

## src/\<pkg\>/utils/seed.py

```python
"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## src/\<pkg\>/utils/logging.py

```python
"""Logging configuration."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
```

---

## src/\<pkg\>/utils/io.py

```python
"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: str | Path,
    **extra: Any,
) -> None:
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
```

---

## scripts/train.py

```python
"""Training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for src layout
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from <pkg>.config import Config
from <pkg>.data.dataset import BaseDataset
from <pkg>.models.base import build_model
from <pkg>.training.trainer import Trainer
from <pkg>.evaluation.evaluator import Evaluator
from <pkg>.utils.logging import setup_logging
from <pkg>.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to experiment override YAML",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.override:
        config = config.merge(args.override)

    setup_logging()
    seed_everything(config.training.seed)

    # TODO: Replace with your dataset and dataloaders
    train_loader = None  # build your DataLoader here
    val_loader = None  # build your DataLoader here

    model = build_model(config.model)
    evaluator = Evaluator(config.evaluation, device=config.training.device)
    trainer = Trainer(model, train_loader, val_loader, config, evaluator)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
```

---

## scripts/evaluate.py

```python
"""Evaluation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from <pkg>.config import Config
from <pkg>.models.base import build_model
from <pkg>.evaluation.evaluator import Evaluator
from <pkg>.utils.io import load_checkpoint
from <pkg>.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    setup_logging()

    model = build_model(config.model)
    load_checkpoint(args.checkpoint, model, device=config.training.device)

    evaluator = Evaluator(config.evaluation, device=config.training.device)

    # TODO: Replace with your dataset and dataloader
    test_loader = None  # build your DataLoader here
    results = evaluator.evaluate(model, test_loader)

    print("Evaluation results:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
```

---

## scripts/preprocess.py

```python
"""Data preprocessing entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from <pkg>.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--input-dir", type=str, required=True, help="Raw data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Processed output directory")
    args = parser.parse_args()

    setup_logging()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing data from {input_dir} to {output_dir}")
    # TODO: Implement preprocessing logic


if __name__ == "__main__":
    main()
```

---

## scripts/predict.py

```python
"""Inference/prediction entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from <pkg>.config import Config
from <pkg>.models.base import build_model
from <pkg>.utils.io import load_checkpoint
from <pkg>.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, default="predictions.pt", help="Output path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    setup_logging()

    model = build_model(config.model)
    load_checkpoint(args.checkpoint, model, device=config.training.device)
    model.eval()

    # TODO: Load input data and run inference
    # predictions = model(input_data)
    # torch.save(predictions, args.output)

    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## tests/\_\_init\_\_.py

```python
```

---

## tests/test_config.py

```python
"""Tests for config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from <pkg>.config import Config


def test_from_yaml_defaults() -> None:
    """Config loads with defaults when YAML is minimal."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        f.flush()
        config = Config.from_yaml(f.name)

    assert config.data.batch_size == 32
    assert config.training.learning_rate == 1e-3
    assert config.training.optimizer == "adam"


def test_from_yaml_override() -> None:
    """Config respects YAML overrides."""
    data = {"data": {"batch_size": 64}, "training": {"learning_rate": 5e-4}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        config = Config.from_yaml(f.name)

    assert config.data.batch_size == 64
    assert config.training.learning_rate == 5e-4
    assert config.training.optimizer == "adam"  # default preserved


def test_merge() -> None:
    """Config.merge applies overrides from a second YAML."""
    base_data = {"training": {"epochs": 100}}
    override_data = {"training": {"epochs": 50, "learning_rate": 1e-4}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base_data, f)
        base_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(override_data, f)
        override_path = f.name

    config = Config.from_yaml(base_path).merge(override_path)
    assert config.training.epochs == 50
    assert config.training.learning_rate == 1e-4


def test_to_yaml_roundtrip() -> None:
    """Config survives a YAML roundtrip."""
    config = Config()
    config.data.batch_size = 128

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        out_path = f.name

    config.to_yaml(out_path)
    loaded = Config.from_yaml(out_path)
    assert loaded.data.batch_size == 128
```

---

## tests/test_dataset.py

```python
"""Tests for dataset module."""

from __future__ import annotations

import pytest

from <pkg>.data.dataset import BaseDataset


def test_base_dataset_not_implemented() -> None:
    """BaseDataset raises NotImplementedError for abstract methods."""
    with pytest.raises(NotImplementedError):
        BaseDataset(data_dir="/tmp", split="train")
```

---

## tests/test_metrics.py

```python
"""Tests for metrics."""

from __future__ import annotations

import torch

from <pkg>.evaluation.metrics import accuracy, get_metric


def test_accuracy_perfect() -> None:
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == 1.0


def test_accuracy_zero() -> None:
    preds = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == 0.0


def test_get_metric() -> None:
    fn = get_metric("accuracy")
    assert fn is accuracy


def test_get_metric_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nonexistent")
```

---

## .gitignore additions

The following entries should be appended to `.gitignore`:

```
# Training data
data/raw/
data/processed/

# Experiment artifacts
checkpoints/
experiments/
notebooks/
```
