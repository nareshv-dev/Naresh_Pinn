"""
scripts/generate_dataset.py

CLI: Generate PyBaMM electrochemical-thermal dataset.

Usage:
  python scripts/generate_dataset.py --config configs/simulation_v0.yaml
  python scripts/generate_dataset.py --config configs/smoke_test.yaml

This script:
  1. Loads the YAML config
  2. Seeds the SQLite manifest
  3. Runs the PyBaMM factorial sweep (SPMe + lumped thermal, Chen2020)
  4. Exports one HDF5 file per simulation run
  5. Registers all runs (pass and fail) in the manifest
  6. Assigns condition-level train/val/test splits in the manifest
  7. Reports a summary of passed/failed runs
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ── Path setup: allow running from project root ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.pybamm_runner import PyBaMMRunner
from src.data.manifest import Manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PyBaMM dataset for Stage I PINN training."
    )
    parser.add_argument(
        "--config", type=str, default="configs/simulation_v0.yaml",
        help="Path to YAML configuration file (relative to project root)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "generate_dataset.log"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), mode="a"),
        ],
    )


def assign_splits(manifest: Manifest, run_metas: list, cfg: dict) -> None:
    """
    Assign condition-level train/validation/test splits.
    Reads split_policy from the YAML config.
    """
    split_policy = cfg["split_policy"]
    config_version = cfg["config_version"]

    # Build lookup: (c_rate, t_amb_K, soc_init) → split
    split_map = {}
    for split_name in ["train", "validation", "test"]:
        for condition in split_policy.get(split_name, []):
            key = (float(condition[0]), float(condition[1]), float(condition[2]))
            split_map[key] = split_name

    for meta in run_metas:
        if not meta.get("quality_passed", False):
            # Failed runs go to a separate "failed" entry but are not split-assigned
            continue
        c_rate = float(meta["c_rate"])
        t_amb = float(meta["ambient_temperature_K"])
        soc = float(meta["initial_soc"])
        key = (c_rate, t_amb, soc)

        split_name = split_map.get(key, "unknown")
        if split_name == "unknown":
            logging.getLogger(__name__).warning(
                "No split policy for condition %s — assigning to 'unknown'", key
            )

        manifest.assign_split(meta["run_id"], split_name, config_version)
        logging.getLogger(__name__).info(
            "  %s -> split=%s", meta["run_id"][:40], split_name
        )


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    log_dir = PROJECT_ROOT / cfg["paths"]["logs_dir"]
    setup_logging(args.log_level, log_dir)
    logger = logging.getLogger("generate_dataset")

    logger.info("=" * 60)
    logger.info("Stage I Dataset Generation")
    logger.info("Config: %s  version=%s", config_path, cfg["config_version"])
    logger.info("=" * 60)

    # ── Open manifest ──────────────────────────────────────────────────────
    db_path = PROJECT_ROOT / cfg["paths"]["sqlite_manifest"]
    manifest = Manifest.open(db_path)

    # Seed reference data
    manifest.seed_parameter_set(
        set_id="Chen2020",
        name="Chen2020 (LG M50)",
        chemistry="NMC/graphite",
        cell_identifier="LG M50",
        source_reference="Chen et al. (2020), J. Electrochem. Soc. 167 080534",
    )
    manifest.seed_protocol(
        protocol_id=cfg["sweep"]["protocol"]["protocol_id"],
        name="Constant Current Discharge",
        protocol_type="CC_discharge",
        c_rate=None,  # per-run values stored in simulation_runs
        voltage_cutoff_V=float(cfg["sweep"]["protocol"]["voltage_cutoff_V"]),
        description=cfg["sweep"]["protocol"]["description"],
    )

    # Register config YAML as artifact
    manifest.register_artifact(
        artifact_type="config_yaml",
        path=str(config_path),
        description="Dataset generation configuration",
        config_version=cfg["config_version"],
    )

    # ── Run PyBaMM sweep ──────────────────────────────────────────────────
    runner = PyBaMMRunner(config=cfg, project_root=PROJECT_ROOT, manifest=manifest)
    run_metas = runner.run_sweep()

    # ── Assign splits ──────────────────────────────────────────────────────
    logger.info("Assigning condition-level splits...")
    assign_splits(manifest, run_metas, cfg)

    # ── Summary ────────────────────────────────────────────────────────────
    n_total = len(run_metas)
    n_passed = sum(1 for m in run_metas if m.get("quality_passed", False))
    n_failed = n_total - n_passed

    logger.info("=" * 60)
    logger.info("Dataset generation complete:")
    logger.info("  Total runs:  %d", n_total)
    logger.info("  Passed:      %d", n_passed)
    logger.info("  Failed:      %d", n_failed)
    logger.info("  Manifest:    %s", db_path)
    logger.info("  Raw HDF5:    %s", PROJECT_ROOT / cfg["paths"]["raw_data_dir"])
    logger.info("=" * 60)

    # Print per-run summary
    for m in run_metas:
        status = "PASS" if m.get("quality_passed", False) else "FAIL"
        logger.info(
            "  %s  C=%.1f  T=%.1f K  n_t=%d  %s",
            status,
            m.get("c_rate", 0),
            m.get("ambient_temperature_K", 0),
            m.get("n_time_steps", 0),
            m.get("run_id", "")[:30],
        )

    manifest.close()

    if n_failed > 0:
        logger.warning("%d runs failed quality checks — see manifest for details.", n_failed)
        return 1  # non-zero exit but don't abort entirely
    return 0


if __name__ == "__main__":
    sys.exit(main())
