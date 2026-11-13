"""Generate the paper-aligned 18-condition dataset and SQLite database."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_experiment.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset = generate_dataset(args.config, force=args.force)
    print(f"Dataset ready: {len(dataset):,} rows, {dataset.condition_id.nunique()} conditions")


if __name__ == "__main__":
    main()
