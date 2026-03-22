"""Training entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for src layout
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.data.dataset import BaseDataset
from event_graph_generation.models.base import build_model
from event_graph_generation.training.trainer import Trainer
from event_graph_generation.evaluation.evaluator import Evaluator
from event_graph_generation.utils.logging import setup_logging
from event_graph_generation.utils.seed import seed_everything

logger = logging.getLogger(__name__)


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

    # DDP initialization
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        import torch
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    config = Config.from_yaml(args.config)
    if args.override:
        config = config.merge(args.override)

    if local_rank >= 0:
        config.training.device = f"cuda:{local_rank}"

    setup_logging()
    seed_everything(config.training.seed)

    if config.model.name == "vjepa_pipeline":
        # V-JEPA Pipeline training
        from torch.utils.data import DataLoader, DistributedSampler

        from event_graph_generation.data.vjepa_dataset import VJEPAEventDataset
        from event_graph_generation.data.vjepa_collator import vjepa_collate_fn
        from event_graph_generation.models.losses import VJEPAEventGraphLoss

        data_dir = Path(config.data.processed_dir)

        train_dataset = VJEPAEventDataset(data_dir=data_dir, split="train")
        val_dataset = VJEPAEventDataset(data_dir=data_dir, split="val")

        train_sampler = (
            DistributedSampler(train_dataset) if local_rank >= 0 else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=vjepa_collate_fn,
            drop_last=True,
            persistent_workers=config.data.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=vjepa_collate_fn,
            persistent_workers=config.data.num_workers > 0,
        )

        loss_weights = {
            "interaction": config.training.loss_weights.interaction,
            "action": config.training.loss_weights.action,
            "agent_ptr": config.training.loss_weights.agent_ptr,
            "target_ptr": config.training.loss_weights.target_ptr,
            "source_ptr": config.training.loss_weights.source_ptr,
            "dest_ptr": config.training.loss_weights.dest_ptr,
            "frame": config.training.loss_weights.frame,
        }
        criterion = VJEPAEventGraphLoss(
            loss_weights=loss_weights,
            num_actions=config.model.num_actions,
        )

        model = build_model(config.model, vjepa_config=config.vjepa)
        evaluator = Evaluator(config.evaluation, device=config.training.device)
        trainer = Trainer(
            model, train_loader, val_loader, config, evaluator, criterion=criterion
        )

        logger.info(f"V-JEPA Pipeline training: {len(train_dataset)} train, {len(val_dataset)} val samples")

    elif config.model.name == "event_decoder":
        # Event Decoder pipeline
        from torch.utils.data import DataLoader, DistributedSampler

        from event_graph_generation.data.event_dataset import EventGraphDataset
        from event_graph_generation.data.event_collator import event_collate_fn
        from event_graph_generation.models.losses import EventGraphLoss

        data_dir = Path(config.data.processed_dir)

        train_dataset = EventGraphDataset(
            data_dir=data_dir,
            split="train",
            max_objects=config.model.max_objects,
        )
        val_dataset = EventGraphDataset(
            data_dir=data_dir,
            split="val",
            max_objects=config.model.max_objects,
        )

        train_sampler = (
            DistributedSampler(train_dataset) if local_rank >= 0 else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=event_collate_fn,
            drop_last=True,
            persistent_workers=config.data.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=event_collate_fn,
            persistent_workers=config.data.num_workers > 0,
        )

        # Build loss with weights from config
        loss_weights = {
            "interaction": config.training.loss_weights.interaction,
            "action": config.training.loss_weights.action,
            "agent_ptr": config.training.loss_weights.agent_ptr,
            "target_ptr": config.training.loss_weights.target_ptr,
            "source_ptr": config.training.loss_weights.source_ptr,
            "dest_ptr": config.training.loss_weights.dest_ptr,
            "frame": config.training.loss_weights.frame,
        }
        criterion = EventGraphLoss(
            loss_weights=loss_weights,
            num_actions=config.model.num_actions,
        )

        model = build_model(config.model)
        evaluator = Evaluator(config.evaluation, device=config.training.device)
        trainer = Trainer(
            model, train_loader, val_loader, config, evaluator, criterion=criterion
        )

        logger.info(f"Event Decoder training: {len(train_dataset)} train, {len(val_dataset)} val samples")
    else:
        # Legacy path
        train_loader = None  # build your DataLoader here
        val_loader = None  # build your DataLoader here

        model = build_model(config.model)
        evaluator = Evaluator(config.evaluation, device=config.training.device)
        trainer = Trainer(model, train_loader, val_loader, config, evaluator)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()

    # DDP cleanup
    if local_rank >= 0:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
