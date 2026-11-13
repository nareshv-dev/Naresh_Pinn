"""Train and evaluate the 3 x 3 x 3 experiment matrix."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.experiment import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_experiment.yaml")
    parser.add_argument("--full", action="store_true", help="Use the 2500-epoch paper protocol")
    args = parser.parse_args()
    metrics = run_matrix(args.config, quick=not args.full)
    print(f"Completed {metrics.run_id.nunique()} runs and {len(metrics):,} metric records")


if __name__ == "__main__":
    main()
