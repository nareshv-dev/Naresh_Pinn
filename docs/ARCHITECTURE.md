# Project architecture

## Scientific flow

```text
Operating condition (C-rate, ambient temperature)
                  |
                  v
        PyBaMM SPMe + lumped thermal
                  |
       concentration c(r,t), T(t), heat
                  |
        +---------+----------+
        |                    |
        v                    v
 concentration branch   temperature branch
 (r,t,C,Tamb), 4x96     (t,C,Tamb), 4x96
        |                    |
 diffusion + flux        thermal residual
        |
        v
 analytical spherical-particle stress operator
        |
 radial stress and tangential stress
```

The proposed model has no trainable stress output head. This enforces concentration-stress consistency by construction. The data-only and prior-style PINN variants retain independent stress heads to reproduce the ablation comparison.

## Data flow

`configs/paper_experiment.yaml` is the single source of experiment settings. `scripts/generate_paper_dataset.py` creates the Parquet dataset and SQLite database. `scripts/run_paper_experiments.py` trains the 27-run matrix, saves checkpoints, predictions and metric records. The notebook exposes the same Python functions for inspection.

## Live software architecture

```text
Selected scenario + optional live controls
                    |
                    v
        deterministic software battery plant
        current, voltage, SOC, surface temperature
                    |
                    +---------------------+
                    |                     |
                    v                     v
      PC-MP-PINN checkpoint       plant thermal forecast
      c(r,t), T(t), stress        intervention comparison
                    |                     |
                    +----------+----------+
                               v
                threshold and domain assessment
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              WebSocket dashboard     SQLite session log
              charts + 3D view        CSV download endpoint
```

The software plant is the simulated sensor source; the PINN is an estimator, not the source of those measurements. This separation makes the measured-versus-estimated comparison meaningful. `configs/digital_twin.yaml` owns scenarios and experimental limits. `src/realtime/` owns plant dynamics, checkpoint inference, orchestration and persistence. FastAPI exposes REST and WebSocket interfaces, while `dashboard/` renders the operator view.

The Three.js scene visualizes one lumped cell temperature and the model’s radial particle state. It deliberately does not claim three-dimensional hotspot localization.

## Result status

The repository includes a completed quick-validation matrix. It verifies all 27 combinations and all nine metrics but is not a substitute for the 2,500-epoch publication run. Use `python scripts/run_paper_experiments.py --full` before copying numbers into a final manuscript.

The live interface uses `outputs/deployment/pc_mp_pinn_live.pt`, a separately labelled deployment fine-tune. Its held-out results are temperature RMSE 0.364 K and concentration relative L2 error 3.15%. These values describe that checkpoint only and are not paper-comparison claims.
