"""
scripts/preprocess.py

CLI: Preprocess HDF5 runs into Parquet index files for training.

Usage:
  python scripts/preprocess.py --config configs/simulation_v0.yaml
  python scripts/preprocess.py --config configs/smoke_test.yaml

This script:
  1. Loads config and manifest
  2. Reads HDF5 files for each split (train, val, test)
  3. Applies nondimensionalization (from config nondim section)
  4. Fits scalers on TRAINING data only
  5. Saves scaler artifact to JSON
  6. Generates labeled sample Parquet files for each split
  7. Generates collocation Parquet files for train split
  8. Writes split assignments to Parquet index files
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import Manifest
from src.data.loaders import load_run_hdf5
from src.preprocessing.nondim import Nondimensionalizer
from src.preprocessing.scalers import ScalerBundle
from src.preprocessing.samplers import LabeledSampler, CollocationSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Stage I dataset.")
    parser.add_argument("--config", type=str, default="configs/simulation_v0.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / "preprocess.log"), mode="a"),
        ],
    )


def make_current_interp(t_star: np.ndarray, I_star: np.ndarray):
    """Return a callable t_star → I_star via linear interpolation."""
    from scipy.interpolate import interp1d
    f = interp1d(t_star, I_star, kind="linear", fill_value="extrapolate")
    return lambda t: float(f(np.clip(t, t_star[0], t_star[-1])))


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log_dir = PROJECT_ROOT / cfg["paths"]["logs_dir"]
    setup_logging(args.log_level, log_dir)
    logger = logging.getLogger("preprocess")

    config_version = cfg["config_version"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stage I Preprocessing — config version: %s", config_version)
    logger.info("=" * 60)

    # ── Open manifest ──────────────────────────────────────────────────────
    db_path = PROJECT_ROOT / cfg["paths"]["sqlite_manifest"]
    manifest = Manifest.open(db_path)

    # ── Initialize nondimensionalizer ──────────────────────────────────────
    nondim = Nondimensionalizer(cfg["nondim"])
    nondim.save(proc_dir / f"nondim_constants_{config_version}.json")

    # ── Training config ────────────────────────────────────────────────────
    tr_cfg = cfg["training"]
    n_lab = int(tr_cfg["n_labeled_samples_per_run"])
    n_int = int(tr_cfg["n_collocation_interior"])
    n_ic = int(tr_cfg["n_collocation_ic"])
    n_bc = int(tr_cfg["n_collocation_bc"])
    sampler_type = str(tr_cfg.get("sampler", "sobol"))
    sampler_seed = int(tr_cfg.get("sampler_seed", 42))

    lab_sampler = LabeledSampler(n_samples_per_run=n_lab, seed=sampler_seed)
    col_sampler = CollocationSampler(
        n_interior=n_int,
        n_ic=n_ic,
        n_bc=n_bc,
        sampler_type=sampler_type,
        seed=sampler_seed,
    )

    # c_s0 normalized: initial SOC × (c_s0/c_s_max)
    # For Chen2020, c_s0_neg_frac ≈ initial SOC. Use 0.8 as default.
    # (The exact value is approximately soc_init * c_s0_neg / c_s_max_neg)
    # We use SOC=1.0 → c* ≈ c_s0_neg / c_s_max_neg
    # From Chen2020 params: c_s0_neg ≈ 28700 mol/m³; c_s_max_neg ≈ 33133
    # c_s0_star ≈ 28700 / 33133 ≈ 0.866
    c_s0_star = 28700.0 / float(cfg["nondim"]["c_s_max_neg_mol_m3"])
    logger.info("IC concentration: c_s0_star = %.4f", c_s0_star)

    # ── Process each split ──────────────────────────────────────────────────
    split_frames: dict = {"train": [], "validation": [], "test": []}
    col_frames: list = []

    for split_name in ["train", "validation", "test"]:
        runs = manifest.get_runs_by_split(split_name, config_version)
        if not runs:
            logger.info("  Split '%s': 0 runs (skipping)", split_name)
            continue
        logger.info("  Split '%s': %d runs", split_name, len(runs))

        run_list_labeled = []
        run_list_col = []

        for run in runs:
            hdf5_path = run.get("hdf5_path")
            if not hdf5_path or not Path(hdf5_path).exists():
                logger.warning("  HDF5 not found: %s — skipping", hdf5_path)
                continue

            arrays = load_run_hdf5(hdf5_path)
            nd_arrays = nondim.nondimensionalize_run(arrays)

            run_list_labeled.append({
                "run_id": run["run_id"],
                "nondim_arrays": nd_arrays,
            })

            if split_name == "train":
                t_max_star = float(nd_arrays["t_star"].max())
                I_star_func = make_current_interp(nd_arrays["t_star"], nd_arrays["I_star"])
                run_list_col.append({
                    "run_id": run["run_id"],
                    "t_max_star": t_max_star,
                    "I_star_func": I_star_func,
                    "T_amb_star": float(nd_arrays["T_amb_star"]),
                    "c_s0_star": c_s0_star,
                })

        df_lab = lab_sampler.sample_multiple_runs(run_list_labeled, split_name)
        split_frames[split_name] = df_lab
        logger.info("    Labeled samples: %d rows", len(df_lab))

        if split_name == "train" and run_list_col:
            df_col = col_sampler.sample_multiple_runs(run_list_col, "train")
            col_frames.append(df_col)
            logger.info("    Collocation points: %d rows", len(df_col))

    # ── Fit scalers on TRAINING data only ─────────────────────────────────
    train_df = split_frames.get("train", pd.DataFrame())
    if len(train_df) == 0:
        logger.error("No training data found — cannot fit scalers!")
        return 1

    INPUT_COLS = ["t_star", "r_star", "I_star", "T_amb_star"]
    TARGET_COLS = ["V_star", "c_star", "T_star"]

    train_inputs = train_df[INPUT_COLS].values.astype(np.float64)
    train_targets = train_df[TARGET_COLS].values.astype(np.float64)

    train_run_ids = sorted(train_df["run_id"].unique().tolist())
    scaler_bundle = ScalerBundle()
    scaler_bundle.fit(
        train_inputs, train_targets,
        fit_note=f"Fitted on train runs: {train_run_ids}"
    )
    scaler_path = PROJECT_ROOT / cfg["paths"]["scaler_artifact"]
    scaler_bundle.save(scaler_path)
    manifest.register_artifact(
        artifact_type="scaler_json",
        path=str(scaler_path),
        description="StandardScaler fitted on training data only",
        config_version=config_version,
    )

    # ── Save labeled Parquet files ─────────────────────────────────────────
    split_parquet_keys = {
        "train": cfg["paths"]["train_index_parquet"],
        "validation": cfg["paths"]["val_index_parquet"],
        "test": cfg["paths"]["test_index_parquet"],
    }

    for split_name, df in split_frames.items():
        if len(df) == 0:
            logger.info("  Split '%s': empty — not saving Parquet", split_name)
            continue
        parquet_path = PROJECT_ROOT / split_parquet_keys[split_name]
        df.to_parquet(str(parquet_path), index=False)
        logger.info("  Saved %s Parquet: %s  (%d rows)", split_name, parquet_path, len(df))
        manifest.register_artifact(
            artifact_type=f"parquet_split_{split_name}",
            path=str(parquet_path),
            description=f"Labeled sample index for {split_name} split",
            config_version=config_version,
        )

    # ── Save collocation Parquet ───────────────────────────────────────────
    if col_frames:
        col_df = pd.concat(col_frames, ignore_index=True)
        col_parquet = proc_dir / f"collocation_train_{config_version}.parquet"
        col_df.to_parquet(str(col_parquet), index=False)
        logger.info("  Saved collocation Parquet: %s  (%d rows)", col_parquet, len(col_df))
        manifest.register_artifact(
            artifact_type="parquet_collocation",
            path=str(col_parquet),
            description="Collocation points for PINN training (train split only)",
            config_version=config_version,
        )

    manifest.close()

    logger.info("=" * 60)
    logger.info("Preprocessing complete. Artifacts saved to %s", proc_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
