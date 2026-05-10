from __future__ import annotations


def test_tc01_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert "request_id" in body["meta"]


def test_tc02_ready_endpoint_reports_registered_models(client):
    response = client.get("/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["n_models"] > 0, "No hay modelos registrados — revisa MODELS_DIR."
    assert data["status"] == "ready"
    assert data["cache_loaded"] >= 0
