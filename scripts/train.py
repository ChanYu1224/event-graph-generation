"""Training entry point."""

from __future__ import annotations

import argparse
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
