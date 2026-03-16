"""Inference/prediction entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.config import Config
from event_graph_generation.models.base import build_model
from event_graph_generation.utils.io import load_checkpoint
from event_graph_generation.utils.logging import setup_logging


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
