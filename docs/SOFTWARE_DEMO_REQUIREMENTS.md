# Software-only digital twin requirements

## Purpose

The demonstrator must behave like a live battery-monitoring system without claiming that physical sensors are connected. A software plant generates telemetry, the trained PINN estimates hidden internal states, and the dashboard streams both through the backend.

## Acceptance requirements

| ID | Requirement | Evidence |
|---|---|---|
| DT-01 | Clearly identify every incoming value as simulated telemetry | Persistent banner and source field in each message |
| DT-02 | Execute a saved PyTorch checkpoint during the live session | Model name, checkpoint and inference latency shown in the UI |
| DT-03 | Stream voltage, current, SOC, surface temperature and ambient temperature | WebSocket `/ws/live` |
| DT-04 | Estimate radial concentration and concentration-coupled particle stress | 25-point radial profile per inference |
| DT-05 | Forecast temperature and stress at 30 s, 60 s and 300 s | Forecast array in live messages |
| DT-06 | Support normal, aggressive, hot-ambient, pulsed-load, cooling-failure and abnormal-heating scenarios | Scenario selector and API validation |
| DT-07 | Support start, pause, resume, reset and speed control | Accessible dashboard controls |
| DT-08 | Persist sessions, telemetry, predictions and alerts | SQLite tables with session export endpoint |
| DT-09 | Display a 3D cylindrical cell and enlarged particle cutaway | Three.js view linked to live temperature, concentration and stress |
| DT-10 | Mark limits as configurable experimental thresholds | Threshold panel and alert explanation |
| DT-11 | Compare continue, reduced-load and enhanced-cooling futures | Scenario comparison API and UI |
| DT-12 | Show model-domain status and measured-versus-estimated distinction | Confidence/domain panel and chart legend |
| DT-13 | Provide a non-3D text/table alternative | Live values, radial table and accessible summaries |
| DT-14 | Work without hardware | Built-in deterministic software telemetry generator |

## Scientific limits

- Temperature is lumped, so the 3D cell colour represents one cell-average temperature. It is not a spatial hotspot map.
- The radial particle view represents normalized electrode-particle radius, not a full physical cell coordinate.
- Cooling interventions use the software plant's thermal model because cooling strength is not an input to the current PINN.
- The paper-comparison matrix remains a quick-validation artifact until the full protocol is completed. The live demonstrator uses a separately labelled deployment checkpoint and reports its held-out metrics; it is not substituted for the paper benchmark.
- Alerts are experimental demonstration thresholds and are not certified BMS commands.
