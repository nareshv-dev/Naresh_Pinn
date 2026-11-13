import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app, manager
from src.realtime.manager import DigitalTwinManager


def test_catalog_discloses_software_source_and_experimental_thresholds():
    catalog = manager.catalog()
    assert "normal" in catalog["scenarios"]
    assert "abnormal_heating" in catalog["scenarios"]
    assert catalog["thresholds"]["temperature_warning_C"] == 45.0
    assert any("no hardware" in message.lower() for message in catalog["disclaimers"])


def test_checkpoint_inference_has_radial_outputs():
    output = manager.engine.predict(elapsed_s=600.0, c_rate=1.0, ambient_c=25.0)
    assert len(output["radius_star"]) == 25
    assert len(output["concentration_mol_m3"]) == 25
    assert len(output["sigma_tangential_MPa"]) == 25
    assert output["checkpoint"].endswith(".pt")
    assert output["inference_ms"] > 0


def test_api_predict_and_catalog():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["simulation_source"] == "software_only"
        assert client.get("/api/live/catalog").status_code == 200
        response = client.post("/api/predict", json={"elapsed_s": 300, "c_rate": 1.25, "ambient_C": 25})
        assert response.status_code == 200
        assert response.json()["confidence"] == 1.0


def test_live_manager_emits_and_persists_one_tick(tmp_path):
    async def run():
        twin = DigitalTwinManager(Path.cwd(), tmp_path / "realtime.sqlite")
        queue = await twin.subscribe()
        started = await twin.start("normal", speed=1.0)
        payload = await asyncio.wait_for(queue.get(), timeout=3.0)
        await twin.stop()
        assert started["status"] == "running"
        assert payload["source"] == "software_simulator"
        assert payload["prediction"]["checkpoint"].endswith(".pt")
        assert twin.store.session_rows(started["session_id"])

    asyncio.run(run())
