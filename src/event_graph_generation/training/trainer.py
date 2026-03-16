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
