"""Training loop."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
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

        # DDP setup
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.is_main = self.rank == 0

        if self.is_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank]
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.evaluator = evaluator
        self._wandb_enabled = _WANDB_AVAILABLE and config.wandb.enabled

        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = build_scheduler(self.optimizer, config.training)

        if criterion is not None:
            self.criterion = criterion.to(config.training.device)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.current_epoch = 0
        self.best_metric = float("-inf")

        # AMP support
        self.use_amp = str(config.training.device).startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = config.training.early_stopping_patience

    @property
    def _raw_model(self) -> nn.Module:
        """Unwrap DDP wrapper if present."""
        return self.model.module if self.is_ddp else self.model

    def train(self) -> None:
        """Run the full training loop."""
        if self._wandb_enabled and self.is_main:
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
            if self._wandb_enabled and self.is_main:
                wandb.finish()

    def _train_loop(self) -> None:
        """Core training loop."""
        device = self.config.training.device
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            train_loss, head_losses = self._train_one_epoch()

            if self.is_main:
                logger.info(f"Epoch {epoch}/{self.config.training.epochs} - loss: {train_loss:.4f}")

            # Build log dict for this epoch
            log_dict: dict[str, float] = {"train/loss": train_loss, "epoch": epoch}
            for head_name, head_val in head_losses.items():
                log_dict[f"train/{head_name}"] = head_val

            # Evaluation (rank 0 only)
            val_loss = None
            if (
                self.is_main
                and self.val_loader is not None
                and self.evaluator is not None
                and (epoch + 1) % self.config.evaluation.eval_every_n_epochs == 0
            ):
                metrics = self.evaluator.evaluate(self._raw_model, self.val_loader)
                logger.info(f"Epoch {epoch} - val metrics: {metrics}")

                val_loss = metrics.get("loss", train_loss)
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v

            # Log all metrics (rank 0 only)
            if self._wandb_enabled and self.is_main:
                wandb.log(log_dict, step=epoch)

            # Early stopping (rank 0 decides, broadcast to all ranks)
            should_stop = False
            if self.is_main:
                check_loss = val_loss if val_loss is not None else train_loss
                if check_loss < self.best_val_loss:
                    self.best_val_loss = check_loss
                    self.patience_counter = 0
                    # Save best model
                    ckpt_dir = Path(self.config.training.checkpoint_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        self._raw_model,
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
                    should_stop = True

            if self.is_ddp:
                stop_tensor = torch.tensor(
                    [1 if should_stop else 0], device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1

            if should_stop:
                break

            # Periodic checkpoint (rank 0 only)
            if self.is_main and (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                ckpt_dir = Path(self.config.training.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    self._raw_model,
                    self.optimizer,
                    epoch,
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

    def _train_one_epoch(self) -> tuple[float, dict[str, float]]:
        """Train for a single epoch. Returns average loss and per-head losses."""
        if self.is_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        head_losses_sum: dict[str, float] = {}
        device = self.config.training.device

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if hasattr(batch, "vjepa_tokens"):
                    # VJEPAEventBatch path
                    batch = batch.to(device)
                    obj_repr, predictions = self.model(batch.vjepa_tokens)
                    loss, head_losses = self.criterion(
                        obj_repr, predictions,
                        batch.gt_events, batch.gt_object_categories,
                    )
                elif hasattr(batch, "object_embeddings"):
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
            self._raw_model,
            self.optimizer,
            device=self.config.training.device,
        )
        self.current_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
