"""Data preprocessing entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_graph_generation.utils.logging import setup_logging

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
