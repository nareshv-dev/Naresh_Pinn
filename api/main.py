"""FastAPI backend for the software-only PC-MP-PINN digital twin."""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.realtime.manager import DigitalTwinManager

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data/processed/paper_v1/battery_pinn.sqlite"
METADATA = ROOT / "data/processed/paper_v1/metadata.json"
DASHBOARD = ROOT / "dashboard"
manager = DigitalTwinManager(ROOT, DATABASE)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await manager.stop()


app = FastAPI(
    title="PC-MP-PINN Software Battery Digital Twin",
    version="2.0.0",
    description="Simulated live telemetry, checkpoint inference, forecasting and experimental alerts.",
    lifespan=lifespan,
)


class StartRequest(BaseModel):
    scenario: str = "normal"
    speed: float = Field(5.0, ge=0.5, le=20.0)


class ControlRequest(BaseModel):
    speed: float | None = Field(None, ge=0.5, le=20.0)
    c_rate: float | None = Field(None, ge=0.1, le=2.5)
    cooling_factor: float | None = Field(None, ge=0.0, le=2.0)


class PredictRequest(BaseModel):
    elapsed_s: float = Field(..., ge=0.0, le=14400.0)
    c_rate: float = Field(..., ge=0.1, le=2.5)
    ambient_C: float = Field(..., ge=-10.0, le=60.0)


def query(sql: str, parameters: tuple = ()) -> list[dict]:
    if not DATABASE.exists():
        raise HTTPException(503, "Run the dataset generator first.")
    with sqlite3.connect(DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "database": DATABASE.exists(), "model_loaded": True, "simulation_source": "software_only"}


@app.get("/api/project")
def project() -> dict:
    metadata = json.loads(METADATA.read_text(encoding="utf-8")) if METADATA.exists() else {}
    return {
        **metadata,
        "title": "Real-Time Physics-Informed Battery Digital Twin",
        "scope": "Software telemetry, PINN internal-state inference and short-horizon experimental forecasting",
        "result_status": "Software-only demonstrator. No physical sensors are connected.",
    }


@app.get("/api/live/catalog")
def live_catalog() -> dict:
    return manager.catalog()


@app.get("/api/live/status")
def live_status() -> dict:
    return manager.state()


@app.post("/api/live/start")
async def live_start(request: StartRequest) -> dict:
    try:
        return await manager.start(request.scenario, request.speed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/live/pause")
async def live_pause() -> dict:
    return await manager.pause()


@app.post("/api/live/resume")
async def live_resume() -> dict:
    return await manager.resume()


@app.post("/api/live/stop")
async def live_stop() -> dict:
    await manager.stop()
    return manager.state()


@app.post("/api/live/control")
def live_control(request: ControlRequest) -> dict:
    return manager.control(request.speed, request.c_rate, request.cooling_factor)


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    return manager.engine.predict(request.elapsed_s, request.c_rate, request.ambient_C)


@app.get("/api/live/compare")
def compare(horizon_s: float = Query(300.0, ge=30.0, le=900.0)) -> list[dict]:
    try:
        return manager.compare(horizon_s)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/live/sessions")
def sessions() -> list[dict]:
    return manager.store.sessions()


@app.get("/api/live/sessions/{session_id}/csv")
def session_csv(session_id: str) -> Response:
    try:
        csv_text = manager.export_csv(session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(csv_text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{session_id}.csv"'})


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket):
    await websocket.accept()
    queue = await manager.subscribe()
    try:
        await websocket.send_json({"type": "connection", "source": "software_simulator", "state": manager.state()})
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(queue)


@app.get("/api/conditions")
def conditions() -> list[dict]:
    return query("SELECT * FROM conditions ORDER BY condition_id")


@app.get("/api/metrics")
def metrics(split: str = Query("test", pattern="^(validation|test)$")) -> list[dict]:
    return query("SELECT e.variant,e.label_budget,e.seed,e.mode,m.metric,m.value FROM metrics m JOIN experiment_runs e ON e.run_id=m.run_id WHERE m.split=? ORDER BY e.variant,e.label_budget,e.seed,m.metric", (split,))


@app.get("/api/profile")
def profile(condition_id: str = "C17") -> list[dict]:
    rows = query("SELECT condition_id,c_rate,ambient_C,time_s,radius_star,concentration_mol_m3,temperature_K,sigma_radial_MPa,sigma_tangential_MPa,voltage_V FROM samples WHERE condition_id=? ORDER BY time_s,radius_star", (condition_id,))
    if not rows:
        raise HTTPException(404, f"Unknown condition: {condition_id}")
    return rows


@app.get("/")
def index():
    return FileResponse(DASHBOARD / "index.html")


app.mount("/dashboard", StaticFiles(directory=DASHBOARD), name="dashboard")
