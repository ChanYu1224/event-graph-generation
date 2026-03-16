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
