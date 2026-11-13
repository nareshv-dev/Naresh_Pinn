"""
src/training/trainer.py

Training loops for EXP-002 (MLP) and EXP-003 (PINN).

Both trainers:
  - Use Adam optimizer
  - Support optional L-BFGS final refinement (only after Adam converges)
  - Log all loss components every N steps
  - Save checkpoints every K epochs
  - Implement early stopping on validation loss
  - Write a JSON training log for later figure generation

MLP trainer:  supervised regression only (data loss)
PINN trainer: joint data + physics losses (5 separate terms tracked)

Inputs/outputs are in normalized (dimensionless) units.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP
from src.models.pinn import PINN

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_training_id(experiment_id: str) -> str:
    return f"{experiment_id}_{uuid.uuid4().hex[:8]}"


# ══════════════════════════════════════════════════════════════════════════════
# MLP Trainer (EXP-002)
# ══════════════════════════════════════════════════════════════════════════════

class MLPTrainer:
    """
    Supervised training loop for the data-only MLP (EXP-002).

    Parameters
    ----------
    model : MLP
    config : dict — training section of YAML config (config['training']['mlp'])
    device : str
    checkpoint_dir : Path
    log_dir : Path
    """

    def __init__(
        self,
        model: MLP,
        config: dict,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("outputs/checkpoints")
        self.log_dir = log_dir or Path("outputs/logs")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_id = _make_training_id(config.get("experiment_id", "EXP-002"))
        self.history: List[Dict] = []

    def train(
        self,
        train_inputs: torch.Tensor,   # shape (N, 4) — on CPU, moved internally
        train_targets: torch.Tensor,  # shape (N, 3)
        val_inputs: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run the full Adam training loop.

        Returns
        -------
        dict with final metrics and training_id.
        """
        cfg = self.cfg
        lr = float(cfg.get("learning_rate", 1e-3))
        batch_size = int(cfg.get("batch_size", 512))
        max_epochs = int(cfg.get("max_epochs", 2000))
        patience = int(cfg.get("early_stopping_patience", 100))
        ckpt_every = int(cfg.get("checkpoint_every_n_epochs", 100))
        log_every = int(cfg.get("log_every_n_steps", 50))

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        train_ds = TensorDataset(
            train_inputs.to(self.device), train_targets.to(self.device)
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_ckpt_path = None
        patience_counter = 0
        start_time = time.time()
        step = 0

        logger.info(
            "MLPTrainer [%s] starting: max_epochs=%d, lr=%.2e, batch=%d",
            self.training_id, max_epochs, lr, batch_size,
        )

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_losses = []

            for batch_in, batch_tgt in loader:
                opt.zero_grad()
                preds = self.model(batch_in)
                loss = nn.functional.mse_loss(preds, batch_tgt)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))
                step += 1

                if step % log_every == 0:
                    logger.debug("step=%d  loss=%.6f", step, loss.item())

            train_loss = float(np.mean(epoch_losses))

            # Validation
            val_loss = None
            if val_inputs is not None and val_targets is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_inputs.to(self.device))
                    val_loss_t = nn.functional.mse_loss(val_preds, val_targets.to(self.device))
                    val_loss = float(val_loss_t.item())

            # Log epoch
            record = {
                "epoch": epoch,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": time.time() - start_time,
            }
            self.history.append(record)

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %4d/%d | train=%.6f | val=%s",
                    epoch, max_epochs,
                    train_loss,
                    f"{val_loss:.6f}" if val_loss is not None else "N/A",
                )

            # Checkpoint
            if epoch % ckpt_every == 0:
                ckpt_path = str(
                    self.checkpoint_dir / f"{self.training_id}_ep{epoch:05d}.pt"
                )
                self.model.save(ckpt_path)

            # Early stopping
            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_val_loss - 1e-8:
                best_val_loss = monitor
                best_ckpt_path = str(
                    self.checkpoint_dir / f"{self.training_id}_best.pt"
                )
                self.model.save(best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)", epoch, patience
                    )
                    break

        elapsed = time.time() - start_time
        # Save training log
        log_path = self.log_dir / f"{self.training_id}_history.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        result = {
            "training_id": self.training_id,
            "final_train_loss": self.history[-1]["train_loss"],
            "best_val_loss": best_val_loss,
            "epochs_run": len(self.history),
            "training_time_s": elapsed,
            "best_checkpoint": best_ckpt_path,
            "log_path": str(log_path),
        }
        logger.info(
            "MLPTrainer done: %.1f s | best_val=%.6f | epochs=%d",
            elapsed, best_val_loss, len(self.history),
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PINN Trainer (EXP-003)
# ══════════════════════════════════════════════════════════════════════════════

class PINNTrainer:
    """
    Training loop for the fixed-weight PINN (EXP-003).

    Manages two parallel data streams:
      1. Labeled data batches (supervised loss)
      2. Collocation point batches (physics losses)

    Parameters
    ----------
    model : PINN
    config : dict — config['training']['pinn']
    c_s0_star : float — initial dimensionless concentration (from config + data)
    device : str
    checkpoint_dir, log_dir : Path
    """

    def __init__(
        self,
        model: PINN,
        config: dict,
        c_s0_star: float = 0.8,
        device: str = "cpu",
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = config
        self.c_s0_star = c_s0_star
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("outputs/checkpoints")
        self.log_dir = log_dir or Path("outputs/logs")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_id = _make_training_id(config.get("experiment_id", "EXP-003"))
        self.history: List[Dict] = []

    def train(
        self,
        # Labeled data tensors
        train_inputs: torch.Tensor,         # (N, 4)
        train_targets: torch.Tensor,        # (N, 3)
        # Collocation tensors (pre-sampled)
        col_interior: torch.Tensor,         # (M_int, 4)  [t*, r*, I*, T_amb*]
        col_ic: torch.Tensor,               # (M_ic, 4)
        col_bc_center: torch.Tensor,        # (M_bc, 4)   r* ≈ 0
        col_bc_surface: torch.Tensor,       # (M_bc, 4)   r* ≈ 1
        # Optional validation data
        val_inputs: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run the full PINN training loop (Adam).

        All collocation tensors are kept on device and sampled in mini-batches
        each epoch (cycling through the full collocation set).

        Returns
        -------
        dict with training result summary.
        """
        cfg = self.cfg
        lr = float(cfg.get("learning_rate", 1e-3))
        batch_size = int(cfg.get("batch_size", 512))
        max_epochs = int(cfg.get("max_epochs", 3000))
        patience = int(cfg.get("early_stopping_patience", 150))
        ckpt_every = int(cfg.get("checkpoint_every_n_epochs", 100))
        log_every = int(cfg.get("log_every_n_steps", 50))

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Move to device
        train_inputs = train_inputs.to(self.device)
        train_targets = train_targets.to(self.device)
        col_int = col_interior.to(self.device)
        col_ic_ = col_ic.to(self.device)
        col_bcc = col_bc_center.to(self.device)
        col_bcs = col_bc_surface.to(self.device)

        # c_s0* tensor for IC loss
        c_s0_val = torch.tensor(self.c_s0_star, dtype=torch.float32, device=self.device)

        # DataLoader for labeled data
        train_ds = TensorDataset(train_inputs, train_targets)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_ckpt_path = None
        patience_counter = 0
        start_time = time.time()
        step = 0

        logger.info(
            "PINNTrainer [%s] starting: max_epochs=%d, lr=%.2e, batch=%d",
            self.training_id, max_epochs, lr, batch_size,
        )
        logger.info("  col_int=%d  col_ic=%d  col_bc=%d", len(col_int), len(col_ic_), len(col_bcc))

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_total = []
            epoch_data = []
            epoch_diff = []
            epoch_therm = []
            epoch_ic_l = []
            epoch_bc_l = []

            # Iterate over labeled batches; cycle collocation batches
            col_int_idx = 0
            col_ic_idx = 0
            col_bc_idx = 0

            for batch_in, batch_tgt in loader:
                opt.zero_grad()

                # ── Sample collocation mini-batches ──────────────────────────
                col_batch = self._sample_col_batch(
                    col_int, col_ic_, col_bcc, col_bcs,
                    batch_size,
                    col_int_idx, col_ic_idx, col_bc_idx,
                    c_s0_val,
                )
                col_int_idx = (col_int_idx + batch_size) % max(len(col_int), 1)
                col_ic_idx = (col_ic_idx + batch_size) % max(len(col_ic_), 1)
                col_bc_idx = (col_bc_idx + batch_size) % max(len(col_bcc), 1)

                # ── Total PINN loss ──────────────────────────────────────────
                L_total, loss_dict = self.model.total_loss(batch_in, batch_tgt, col_batch)

                L_total.backward()
                opt.step()

                epoch_total.append(loss_dict["total"])
                epoch_data.append(loss_dict["data"])
                epoch_diff.append(loss_dict["diffusion"])
                epoch_therm.append(loss_dict["thermal"])
                epoch_ic_l.append(loss_dict["ic"])
                epoch_bc_l.append(loss_dict["bc"])
                step += 1

            # Epoch summary
            train_total = float(np.mean(epoch_total))
            val_loss = None
            if val_inputs is not None and val_targets is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_inputs.to(self.device))
                    vl = nn.functional.mse_loss(val_preds, val_targets.to(self.device))
                    val_loss = float(vl.item())

            record = {
                "epoch": epoch,
                "step": step,
                "train_total": train_total,
                "train_data": float(np.mean(epoch_data)),
                "train_diffusion": float(np.mean(epoch_diff)),
                "train_thermal": float(np.mean(epoch_therm)),
                "train_ic": float(np.mean(epoch_ic_l)),
                "train_bc": float(np.mean(epoch_bc_l)),
                "val_loss": val_loss,
                "elapsed_s": time.time() - start_time,
            }
            self.history.append(record)

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %4d/%d | total=%.5f data=%.5f diff=%.5f therm=%.5f "
                    "ic=%.5f bc=%.5f | val=%s",
                    epoch, max_epochs,
                    record["train_total"], record["train_data"],
                    record["train_diffusion"], record["train_thermal"],
                    record["train_ic"], record["train_bc"],
                    f"{val_loss:.6f}" if val_loss is not None else "N/A",
                )

            if epoch % ckpt_every == 0:
                ckpt_path = str(
                    self.checkpoint_dir / f"{self.training_id}_ep{epoch:05d}.pt"
                )
                self.model.save(ckpt_path)

            monitor = val_loss if val_loss is not None else train_total
            if monitor < best_val_loss - 1e-8:
                best_val_loss = monitor
                best_ckpt_path = str(
                    self.checkpoint_dir / f"{self.training_id}_best.pt"
                )
                self.model.save(best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        elapsed = time.time() - start_time
        log_path = self.log_dir / f"{self.training_id}_history.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        result = {
            "training_id": self.training_id,
            "final_train_total": self.history[-1]["train_total"],
            "best_val_loss": best_val_loss,
            "epochs_run": len(self.history),
            "training_time_s": elapsed,
            "best_checkpoint": best_ckpt_path,
            "log_path": str(log_path),
        }
        logger.info(
            "PINNTrainer done: %.1f s | best_val=%.6f | epochs=%d",
            elapsed, best_val_loss, len(self.history),
        )
        return result

    def _sample_col_batch(
        self,
        col_int: torch.Tensor,
        col_ic: torch.Tensor,
        col_bcc: torch.Tensor,
        col_bcs: torch.Tensor,
        n: int,
        idx_int: int,
        idx_ic: int,
        idx_bc: int,
        c_s0_val: torch.Tensor,
    ) -> Dict:
        """
        Extract mini-batches from pre-sampled collocation tensors.
        Each tensor is [t*, r*, I*, T_amb*], shape (M, 4).
        Returns a dict with individual coordinate tensors that require grad.
        """
        def _get_batch(tensor: torch.Tensor, idx: int) -> torch.Tensor:
            if len(tensor) == 0:
                return tensor
            idx = idx % len(tensor)
            end = min(idx + n, len(tensor))
            return tensor[idx:end]

        int_batch = _get_batch(col_int, idx_int)
        ic_batch = _get_batch(col_ic, idx_ic)
        bcc_batch = _get_batch(col_bcc, idx_bc)
        bcs_batch = _get_batch(col_bcs, idx_bc)

        def _split_coords(batch: torch.Tensor):
            """Split (M, 4) → individual (M, 1) tensors with grad."""
            if len(batch) == 0:
                # Return tiny placeholder tensors
                d = torch.zeros(1, 1, device=self.device, requires_grad=True)
                return d, d.clone(), d.clone(), d.clone()
            t = batch[:, 0:1].detach().requires_grad_(True)
            r = batch[:, 1:2].detach().requires_grad_(True)
            I = batch[:, 2:3].detach()
            T_amb = batch[:, 3:4].detach()
            return t, r, I, T_amb

        t_int, r_int, I_int, T_int = _split_coords(int_batch)
        t_ic, r_ic, I_ic, T_ic = _split_coords(ic_batch)
        t_bcc, r_bcc, I_bcc, T_bcc = _split_coords(bcc_batch)
        t_bcs, r_bcs, I_bcs, T_bcs = _split_coords(bcs_batch)

        # IC concentration reference
        n_ic = len(ic_batch) if len(ic_batch) > 0 else 1
        c_s0_batch = c_s0_val.expand(n_ic, 1)

        return {
            "t_int": t_int, "r_int": r_int, "I_int": I_int, "T_amb_int": T_int,
            "t_ic": t_ic, "r_ic": r_ic, "I_ic": I_ic, "T_amb_ic": T_ic,
            "c_s0_ic": c_s0_batch,
            "t_bc_c": t_bcc, "r_bc_c": r_bcc, "I_bc_c": I_bcc, "T_amb_bc_c": T_bcc,
            "t_bc_s": t_bcs, "r_bc_s": r_bcs, "I_bc_s": I_bcs, "T_amb_bc_s": T_bcs,
        }
