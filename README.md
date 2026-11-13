# Real-Time PC-MP-PINN Battery Digital Twin

Software-only final-year-project demonstrator built around **A Physics-Consistent Electrochemical-Thermal PINN with Concentration-Coupled Particle Stress for Lithium-Ion Battery Surrogate Modeling**.

The project predicts negative-particle lithium concentration and lumped cell temperature from radius, time, C-rate and ambient temperature. The proposed model computes radial and tangential particle stress from concentration with an analytical spherical-particle operator. A built-in software battery plant streams realistic telemetry through FastAPI and WebSockets, allowing the trained checkpoint to operate as a live digital twin without physical hardware.

## Final-year demonstration

The main demonstration now provides:

- Six repeatable software driving and fault scenarios.
- Simulated voltage, current, SOC, ambient and surface-temperature telemetry.
- Actual PyTorch checkpoint inference on every live tick.
- A separately labelled live-deployment checkpoint with held-out temperature RMSE and concentration relative-L2 metrics displayed in the interface; this does not replace the paper-comparison matrix.
- Twenty-five-point concentration and particle-stress profiles.
- Thirty-second, one-minute and five-minute forecasts.
- Continue, reduced-load and enhanced-cooling comparisons.
- A Three.js cylindrical cell and enlarged particle cutaway.
- Configurable, explicitly experimental warning thresholds.
- SQLite session history and telemetry CSV export.
- Model-domain confidence and measured-versus-estimated labelling.

All incoming telemetry is explicitly labelled as simulated. Temperature is lumped and the 3D view does not claim spatial hotspot localization. See [software demonstration requirements](docs/SOFTWARE_DEMO_REQUIREMENTS.md).

## What is complete

- 18 distinct PyBaMM SPMe + lumped-thermal simulations using Chen2020 parameters.
- Exact condition split: 12 train, 4 validation and 2 held-out test conditions.
- 100 time points × 25 radial points per condition = 45,000 rows.
- Three comparison variants: data-only, prior-style PINN and proposed PC-MP-PINN.
- Four hidden layers × 96 tanh units, hard initial conditions, diffusion, center, flux and thermal residuals.
- Analytical concentration-to-stress coupling for the proposed model.
- 3 label budgets × 3 seeds × 3 model variants = 27 experiment runs.
- Nine paper metrics, checkpoints, predictions, SQLite data, Excel report, notebook, FastAPI and responsive dashboard.

The included metrics are a **quick-validation benchmark** (60 epochs) that verifies the whole pipeline. The full configuration keeps the paper protocol: Adam at 1e-3, 2,500 maximum epochs, patience 350, gradient clipping 10, 900 interior and 120 boundary points per training condition. Run the full protocol before using numerical results in a publication.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Python 3.11 or 3.12 is supported.

## Run the project

```powershell
# Recreate the 45,000-row PyBaMM dataset and SQLite database
python scripts/generate_paper_dataset.py --force

# Quick 27-run verification matrix
python scripts/run_paper_experiments.py

# Full paper protocol (long CPU run; a GPU is recommended)
python scripts/run_paper_experiments.py --full

# Start the live API, WebSocket simulator and 3D digital twin
python scripts/serve_dashboard.py
```

Open <http://127.0.0.1:8000>. Select a scenario and press **Start**. API documentation is available at <http://127.0.0.1:8000/docs>.

Important live endpoints:

- `POST /api/live/start`: start a software scenario.
- `POST /api/live/control`: change current, cooling or speed.
- `WS /ws/live`: receive live telemetry, inference, forecasts and alerts.
- `POST /api/predict`: execute the saved checkpoint for one operating point.
- `GET /api/live/compare`: compare five-minute interventions.
- `GET /api/live/sessions/{id}/csv`: export a recorded session.

## Notebook

Open `notebooks/PC_MP_PINN_Training_Evaluation.ipynb`. It covers dataset generation, model training, validation, held-out testing, all nine metrics and a comparison plot. Set `FULL_PROTOCOL = True` inside the notebook to execute the publication protocol.

## Data access

- `data/processed/paper_v1/dataset.parquet`: efficient 45,000-row research dataset.
- `data/processed/paper_v1/battery_pinn.sqlite`: tables `samples`, `conditions`, `experiment_runs` and `metrics`.
- `outputs/battery_pinn_report/battery_pinn_dataset.xlsx`: formatted Excel workbook with summary, conditions, data, metrics and a data dictionary.
- `/api/conditions`, `/api/metrics` and `/api/profile`: browser-friendly JSON endpoints.

## Structure

```text
api/                         FastAPI service
configs/paper_experiment.yaml single experiment specification
configs/digital_twin.yaml    scenarios and experimental thresholds
dashboard/                   deterministic responsive showcase UI
data/processed/paper_v1/     Parquet, SQLite and metadata
docs/ARCHITECTURE.md         scientific and software architecture
notebooks/                   training/evaluation notebook
outputs/paper_v1/            checkpoints, predictions and metrics
outputs/deployment/          evaluated live-inference checkpoint
scripts/                     dataset, experiment and website entry points
src/paper/                   models, physics, stress, data and evaluation
src/realtime/                software plant, inference, sessions and streaming
tests/                       legacy and paper-pipeline verification
```

## Scientific boundary

This is a surrogate-modeling project for negative-particle concentration, lumped temperature and concentration-induced particle stress. It does not estimate degradation, state of health, remaining useful life or cell-replacement decisions. Those were earlier ideas, but they are outside the attached paper’s final scope.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the detailed flow and result-status rules, and [docs/FINAL_YEAR_DEMO_GUIDE.md](docs/FINAL_YEAR_DEMO_GUIDE.md) for the presentation sequence.
