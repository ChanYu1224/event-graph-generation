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
    # Event Decoder parameters
    d_model: int = 256
    nhead: int = 8
    num_object_encoder_layers: int = 3
    num_context_encoder_layers: int = 3
    num_event_decoder_layers: int = 4
    num_event_queries: int = 20
    max_objects: int = 30
    dropout: float = 0.1
    d_geo: int = 12
    d_pair: int = 7
    num_actions: int = 13
    embedding_dim: int = 256


@dataclass
class SAM3Config:
    model_size: str = "large"
    device: str = "cuda"
    batch_size: int = 1
    score_threshold: float = 0.3
    embedding_dim: int = 256
    concept_prompts: list[str] = field(
        default_factory=lambda: ["person", "wrench", "screwdriver", "drawer", "workbench"]
    )
    output_dir: str = "data/sam3_outputs"


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 4096
    temperature: float = 0.1
    do_sample: bool = True
    thinking: bool = False
    clip_length: int = 16
    clip_stride: int = 8
    target_fps: float = 1.0
    max_retries: int = 3
    output_dir: str = "data/annotations"
    quantization: str = "none"  # "none", "4bit", "8bit"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    backend: str = "transformers"  # "transformers" or "vllm"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    max_num_seqs: int = 5
    limit_mm_per_prompt: int = 16


@dataclass
class EventDecoderConfig:
    """Training-specific config for Event Decoder loss weights."""
    interaction: float = 2.0
    action: float = 1.0
    agent_ptr: float = 1.0
    target_ptr: float = 1.0
    source_ptr: float = 0.5
    dest_ptr: float = 0.5
    frame: float = 0.5


@dataclass
class InferenceConfig:
    sam3: SAM3Config = field(default_factory=SAM3Config)
    target_fps: float = 1.0
    clip_length: int = 16
    clip_stride: int = 8
    checkpoint_path: str = "data/checkpoints/best.pt"
    confidence_threshold: float = 0.5
    device: str = "cuda"
    nms_frame_threshold: int = 3
    output_dir: str = "output/event_graphs"
    output_format: str = "json"


@dataclass
class SchedulerParams:
    T_max: int = 100
    warmup_epochs: int = 0
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_params: SchedulerParams = field(default_factory=SchedulerParams)
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 0
    loss_weights: EventDecoderConfig = field(default_factory=EventDecoderConfig)


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
    sam3: SAM3Config = field(default_factory=SAM3Config)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

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
            training_d = d["training"].copy()
            if "scheduler_params" in training_d:
                sp = _update_dataclass(
                    cfg.training.scheduler_params, training_d.pop("scheduler_params")
                )
                cfg.training = _update_dataclass(cfg.training, training_d)
                cfg.training.scheduler_params = sp
            else:
                cfg.training = _update_dataclass(cfg.training, training_d)
            if "loss_weights" in d["training"]:
                cfg.training.loss_weights = _update_dataclass(
                    cfg.training.loss_weights, d["training"]["loss_weights"]
                )
        if "evaluation" in d:
            cfg.evaluation = _update_dataclass(cfg.evaluation, d["evaluation"])
        if "wandb" in d:
            cfg.wandb = _update_dataclass(cfg.wandb, d["wandb"])
        if "sam3" in d:
            cfg.sam3 = _update_dataclass(cfg.sam3, d["sam3"])
        if "vlm" in d:
            cfg.vlm = _update_dataclass(cfg.vlm, d["vlm"])
        if "inference" in d:
            inference_d = d["inference"].copy()
            if "sam3" in inference_d:
                cfg.inference.sam3 = _update_dataclass(
                    cfg.inference.sam3, inference_d.pop("sam3")
                )
            cfg.inference = _update_dataclass(cfg.inference, inference_d)
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
