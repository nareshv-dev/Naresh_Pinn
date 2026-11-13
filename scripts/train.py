"""
scripts/train.py

CLI: Train EXP-002 (MLP) or EXP-003 (PINN).

Usage:
  python scripts/train.py --config configs/simulation_v0.yaml --experiment EXP-002
  python scripts/train.py --config configs/simulation_v0.yaml --experiment EXP-003
  python scripts/train.py --config configs/smoke_test.yaml --experiment EXP-002-smoke

This script:
  1. Loads config and preprocessed Parquet files
  2. Builds the model (MLP or PINN)
  3. Trains with Adam optimizer
  4. Saves best checkpoint
  5. Registers training run and metrics in SQLite manifest
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import MLP
from src.models.pinn import PINN
from src.training.trainer import MLPTrainer, PINNTrainer
from src.preprocessing.nondim import Nondimensionalizer
from src.data.manifest import Manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage I models.")
    parser.add_argument("--config", type=str, default="configs/simulation_v0.yaml")
    parser.add_argument(
        "--experiment", type=str, required=True,
        help="Experiment ID: EXP-002 (MLP) or EXP-003 (PINN)"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / "train.log"), mode="a"),
        ],
    )


def load_labeled_split(parquet_path: Path) -> tuple:
    """Load a Parquet labeled split and return (inputs, targets) as numpy arrays."""
    INPUT_COLS = ["t_star", "r_star", "I_star", "T_amb_star"]
    TARGET_COLS = ["V_star", "c_star", "T_star"]
    df = pd.read_parquet(str(parquet_path))
    X = df[INPUT_COLS].values.astype(np.float32)
    Y = df[TARGET_COLS].values.astype(np.float32)
    return X, Y


def load_collocation(col_parquet_path: Path, c_s0_star: float) -> dict:
    """
    Load collocation Parquet and split into interior / IC / BC tensors.
    Returns a dict of torch tensors ready for PINNTrainer.
    """
    df = pd.read_parquet(str(col_parquet_path))
    COORD_COLS = ["t_star", "r_star", "I_star", "T_amb_star"]

    def get_split(pt_type):
        sub = df[df["point_type"] == pt_type][COORD_COLS].values.astype(np.float32)
        return torch.tensor(sub, dtype=torch.float32)

    col_int = get_split("interior")
    col_ic = get_split("ic")
    col_bcc = get_split("bc_center")
    col_bcs = get_split("bc_surface")

    logging.getLogger("train").info(
        "Collocation: interior=%d  ic=%d  bc_center=%d  bc_surface=%d",
        len(col_int), len(col_ic), len(col_bcc), len(col_bcs),
    )
    return col_int, col_ic, col_bcc, col_bcs


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log_dir = PROJECT_ROOT / cfg["paths"]["logs_dir"]
    setup_logging(args.log_level, log_dir)
    logger = logging.getLogger("train")

    config_version = cfg["config_version"]
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    device = args.device
    exp_id = args.experiment

    logger.info("=" * 60)
    logger.info("Training %s | config=%s | device=%s", exp_id, config_version, device)
    logger.info("=" * 60)

    # Determine model type from experiment ID
    is_pinn = "003" in exp_id or exp_id.upper().startswith("PINN")
    model_key = "pinn" if is_pinn else "mlp"
    model_cfg = cfg["training"][model_key]

    # ── Load data ──────────────────────────────────────────────────────────
    train_parquet = PROJECT_ROOT / cfg["paths"]["train_index_parquet"]
    val_parquet = PROJECT_ROOT / cfg["paths"]["val_index_parquet"]

    if not train_parquet.exists():
        logger.error("Train Parquet not found: %s — run preprocess.py first", train_parquet)
        return 1

    X_train, Y_train = load_labeled_split(train_parquet)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    logger.info("Train: %d samples", len(X_train_t))

    X_val_t = Y_val_t = None
    if val_parquet.exists():
        X_val, Y_val = load_labeled_split(val_parquet)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
        logger.info("Val: %d samples", len(X_val_t))

    # ── Build model ────────────────────────────────────────────────────────
    hidden_dims = list(model_cfg["hidden_dims"])
    activation = str(model_cfg.get("activation", "tanh"))
    input_dim = int(model_cfg.get("input_dim", 4))
    output_dim = int(model_cfg.get("output_dim", 3))

    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]
    tr_cfg_ext = {**model_cfg,
                  "checkpoint_every_n_epochs": cfg["training"].get("checkpoint_every_n_epochs", 100),
                  "log_every_n_steps": cfg["training"].get("log_every_n_steps", 50)}

    if not is_pinn:
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        trainer = MLPTrainer(
            model=model,
            config=tr_cfg_ext,
            device=device,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )
        result = trainer.train(X_train_t, Y_train_t, X_val_t, Y_val_t)

    else:
        loss_weights = dict(model_cfg.get("loss_weights", {}))
        model = PINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            loss_weights=loss_weights,
        )

        # c_s0_star — from nondim + initial SOC
        c_s0_star = 28700.0 / float(cfg["nondim"]["c_s_max_neg_mol_m3"])

        # Load collocation
        col_parquet = proc_dir / f"collocation_train_{config_version}.parquet"
        if not col_parquet.exists():
            logger.error("Collocation Parquet not found: %s", col_parquet)
            return 1

        col_int, col_ic, col_bcc, col_bcs = load_collocation(col_parquet, c_s0_star)

        trainer = PINNTrainer(
            model=model,
            config=tr_cfg_ext,
            c_s0_star=c_s0_star,
            device=device,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )
        result = trainer.train(
            X_train_t, Y_train_t,
            col_int, col_ic, col_bcc, col_bcs,
            X_val_t, Y_val_t,
        )

    # ── Register training run in manifest ──────────────────────────────────
    db_path = PROJECT_ROOT / cfg["paths"]["sqlite_manifest"]
    with Manifest.open(db_path) as manifest:
        training_id = result["training_id"]
        manifest.register_training_run({
            "id": training_id,
            "experiment_id": exp_id,
            "model_class": "PINN" if is_pinn else "MLP",
            "config_version": config_version,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "optimizer": model_cfg.get("optimizer", "Adam"),
            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),
            "max_epochs": int(model_cfg.get("max_epochs", 2000)),
            "n_parameters": model.count_parameters() if hasattr(model, "count_parameters") else 0,
            "checkpoint_path": result.get("best_checkpoint"),
            "started_at": None,  # logged by trainer internally
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "done",
        })

    logger.info("=" * 60)
    logger.info("Training complete:")
    logger.info("  Training ID: %s", result["training_id"])
    logger.info("  Epochs run:  %d", result["epochs_run"])
    logger.info("  Best val loss: %.6f", result["best_val_loss"])
    logger.info("  Time: %.1f s", result["training_time_s"])
    logger.info("  Checkpoint: %s", result.get("best_checkpoint"))
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
