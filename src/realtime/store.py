"""SQLite persistence for digital-twin sessions."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class RealtimeStore:
    def __init__(self, database: Path):
        self.database = database
        self._create_schema()

    def _connect(self):
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS live_sessions(
              session_id TEXT PRIMARY KEY, scenario TEXT NOT NULL, status TEXT NOT NULL,
              started_at TEXT NOT NULL, ended_at TEXT, settings_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_telemetry(
              id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
              wall_time TEXT NOT NULL, simulated_time_s REAL NOT NULL, source TEXT NOT NULL,
              current_A REAL, voltage_V REAL, soc_percent REAL, ambient_C REAL,
              measured_temperature_C REAL, predicted_temperature_C REAL,
              heat_generation_W REAL, peak_stress_MPa REAL, peak_stress_radius REAL,
              confidence REAL, risk_level TEXT, payload_json TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES live_sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_live_session_sequence ON live_telemetry(session_id, sequence);
            """)

    def start_session(self, session_id: str, scenario: str, started_at: str, settings: dict) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO live_sessions(session_id,scenario,status,started_at,settings_json) VALUES(?,?,?,?,?)", (session_id, scenario, "running", started_at, json.dumps(settings)))

    def save_tick(self, payload: dict) -> None:
        telemetry, prediction = payload["telemetry"], payload["prediction"]
        with self._connect() as connection:
            connection.execute("""INSERT INTO live_telemetry(
              session_id,sequence,wall_time,simulated_time_s,source,current_A,voltage_V,soc_percent,
              ambient_C,measured_temperature_C,predicted_temperature_C,heat_generation_W,
              peak_stress_MPa,peak_stress_radius,confidence,risk_level,payload_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                payload["session_id"], payload["sequence"], payload["wall_time"], payload["simulated_time_s"],
                payload["source"], telemetry["current_A"], telemetry["voltage_V"], telemetry["soc_percent"],
                telemetry["ambient_C"], telemetry["surface_temperature_C"], prediction["temperature_C"],
                telemetry["heat_generation_W"], prediction["peak_tensile_stress_MPa"],
                prediction["peak_stress_radius_star"], prediction["confidence"], payload["risk"]["level"],
                json.dumps(payload),
            ))

    def set_status(self, session_id: str, status: str, ended_at: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE live_sessions SET status=?,ended_at=COALESCE(?,ended_at) WHERE session_id=?", (status, ended_at, session_id))

    def sessions(self, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM live_sessions ORDER BY started_at DESC LIMIT ?", (limit,))]

    def session_rows(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM live_telemetry WHERE session_id=? ORDER BY sequence", (session_id,))]
