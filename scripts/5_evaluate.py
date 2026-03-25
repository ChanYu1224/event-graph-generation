"""Evaluation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.models.base import build_model
from event_graph_generation.evaluation.evaluator import Evaluator
from event_graph_generation.utils.io import load_checkpoint
from event_graph_generation.utils.logging import setup_logging


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
