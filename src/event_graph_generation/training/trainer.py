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

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Trainer:
    """Custom training loop with AMP, early stopping, and per-head loss logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: Config,
        evaluator: Evaluator | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.evaluator = evaluator
        self._wandb_enabled = _WANDB_AVAILABLE and config.wandb.enabled

        self.optimizer = build_optimizer(model, config.training)
        self.scheduler = build_scheduler(self.optimizer, config.training)

        if criterion is not None:
            self.criterion = criterion.to(config.training.device)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.current_epoch = 0
        self.best_metric = float("-inf")

        # AMP support
        self.use_amp = config.training.device == "cuda" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = config.training.early_stopping_patience

    def train(self) -> None:
        """Run the full training loop."""
        if self._wandb_enabled:
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                config=self.config.to_dict(),
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
            )

        try:
            self._train_loop()
        finally:
            if self._wandb_enabled:
                wandb.finish()

    def _train_loop(self) -> None:
        """Core training loop."""
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            train_loss, head_losses = self._train_one_epoch()
            logger.info(f"Epoch {epoch}/{self.config.training.epochs} - loss: {train_loss:.4f}")

            # Build log dict for this epoch
            log_dict: dict[str, float] = {"train/loss": train_loss, "epoch": epoch}
            for head_name, head_val in head_losses.items():
                log_dict[f"train/{head_name}"] = head_val

            # Evaluation
            val_loss = None
            if (
                self.val_loader is not None
                and self.evaluator is not None
                and (epoch + 1) % self.config.evaluation.eval_every_n_epochs == 0
            ):
                metrics = self.evaluator.evaluate(self.model, self.val_loader)
                logger.info(f"Epoch {epoch} - val metrics: {metrics}")

                val_loss = metrics.get("loss", train_loss)
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v

            # Log all metrics in a single wandb.log call
            if self._wandb_enabled:
                wandb.log(log_dict, step=epoch)

            # Early stopping
            check_loss = val_loss if val_loss is not None else train_loss
            if check_loss < self.best_val_loss:
                self.best_val_loss = check_loss
                self.patience_counter = 0
                # Save best model
                ckpt_dir = Path(self.config.training.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    ckpt_dir / "best.pt",
                )
                logger.info(f"Saved best model at epoch {epoch} with loss {check_loss:.4f}")
            else:
                self.patience_counter += 1

            if (
                self.early_stopping_patience > 0
                and self.patience_counter >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience {self.early_stopping_patience} exhausted)"
                )
                break

            # Periodic checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                ckpt_dir = Path(self.config.training.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

    def _train_one_epoch(self) -> tuple[float, dict[str, float]]:
        """Train for a single epoch. Returns average loss and per-head losses."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        head_losses_sum: dict[str, float] = {}
        device = self.config.training.device

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if hasattr(batch, "object_embeddings"):
                    # EventBatch path
                    batch = batch.to(device)
                    predictions = self.model(
                        batch.object_embeddings,
                        batch.object_temporal,
                        batch.pairwise,
                        batch.object_mask,
                    )
                    loss, head_losses = self.criterion(
                        predictions, batch.gt_events, batch.object_mask
                    )
                else:
                    # Legacy path
                    outputs = self.model(batch.inputs.to(device))
                    loss = self.criterion(outputs, batch.targets.to(device))
                    head_losses = {}

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.training.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                self.optimizer.step()

            total_loss += loss.item()
            for k, v in head_losses.items():
                head_losses_sum[k] = head_losses_sum.get(k, 0.0) + v
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_head_losses = {
            k: v / max(num_batches, 1) for k, v in head_losses_sum.items()
        }
        return avg_loss, avg_head_losses

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
