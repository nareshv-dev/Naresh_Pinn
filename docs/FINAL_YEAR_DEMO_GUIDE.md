# Final-year demonstration guide

## One-sentence project statement

This project is a software-only lithium-ion battery digital twin that combines a simulated battery plant with a physics-consistent neural surrogate to estimate internal lithium concentration, particle stress and temperature in real time, forecast risk and compare interventions.

## What to demonstrate in six minutes

1. Open the dashboard and point to **Software simulation — No physical hardware connected**.
2. Start **Normal urban load** at 5×. Explain that voltage, current, SOC and surface temperature are simulated sensor values, while temperature, concentration and stress marked **estimated** come from the saved PINN checkpoint.
3. Show the 3D cell and enlarged particle. Explain that colour represents lumped temperature and radial concentration; it is not a cell-hotspot scan.
4. Show the temperature timeline, radial profile and inference latency. Mention the displayed held-out metrics: 0.364 K temperature RMSE and 3.15% concentration relative L2 error.
5. Open **Compare interventions** and explain the five-minute continue, reduce-load and enhanced-cooling alternatives.
6. Reset, choose **Abnormal heating injection** at 20× and start. Show the repeatable experimental critical alert and recommendation. State that 45 °C and 55 °C are configurable demonstration thresholds, not certified BMS limits.
7. Export the session CSV, then mention that the research dataset is also available in Excel and SQLite.

## Questions you should be ready to answer

- **Where does the data come from?** The training dataset comes from PyBaMM SPMe simulations with a lumped thermal model and Chen2020 parameters. Live telemetry comes from a deterministic software plant so the demonstration requires no hardware.
- **Why is it a digital twin?** A dynamic virtual plant produces the observable state; a trained model continuously estimates hidden internal states, assesses risk, forecasts future behavior and records each synchronized session.
- **What makes the model physics-informed?** Its loss includes diffusion, center, surface-flux and thermal residuals. The proposed architecture derives stress analytically from concentration instead of learning an unrelated stress output.
- **What is actually real time?** Backend inference, WebSocket transmission, threshold evaluation, visualization, control changes and SQLite recording all happen continuously while the scenario runs. The input source is clearly simulated.
- **Why not claim hotspot detection?** The thermal target is lumped cell temperature, so the model has no spatial cell-temperature field from which a hotspot coordinate could be inferred.
- **Are the safety limits certified?** No. They are explicit experimental thresholds for academic demonstration.

## Run checklist

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/serve_dashboard.py
```

Then open <http://127.0.0.1:8000>. Keep the terminal visible in case you want to show WebSocket/API activity. API documentation is at <http://127.0.0.1:8000/docs>.

Before the presentation, confirm:

- the checkpoint name and held-out metrics appear in the header;
- normal mode stays below experimental warning limits;
- abnormal heating produces an experimental alert;
- the browser can download a session CSV;
- the Excel report opens at `outputs/battery_pinn_report/battery_pinn_dataset.xlsx`;
- you describe the 27-run matrix as quick validation unless you have completed the full 2,500-epoch protocol.
