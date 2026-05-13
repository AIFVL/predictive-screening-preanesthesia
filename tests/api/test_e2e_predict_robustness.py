from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _ExplodingModel:
    def predict_proba(self, _df):
        raise RuntimeError("simulated model failure")


@pytest.fixture
def lenient_client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_tc22_model_predict_failure_returns_500(
    lenient_client, predict_url, low_risk_features, app, monkeypatch
):
    registry = app.state.registry
    monkeypatch.setattr(registry, "load_model", lambda **_kw: _ExplodingModel())

    response = lenient_client.post(predict_url, json={"features": low_risk_features})

    assert response.status_code == 500, response.text


def test_tc23_health_and_predict_recover_after_model_failure(
    client, predict_url, low_risk_features, app, monkeypatch
):
    registry = app.state.registry

    with TestClient(app, raise_server_exceptions=False) as failing_client:
        monkeypatch.setattr(registry, "load_model", lambda **_kw: _ExplodingModel())
        failing_client.post(predict_url, json={"features": low_risk_features})

    monkeypatch.undo()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["data"]["status"] == "ok"

    ok_response = client.post(predict_url, json={"features": low_risk_features})
    assert ok_response.status_code == 200, ok_response.text
    assert ok_response.json()["success"] is True
