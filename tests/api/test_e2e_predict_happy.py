from __future__ import annotations


_RISK_LEVELS = {"low", "moderate", "elevated", "high"}


def _assert_predict_response_shape(body: dict, manifest) -> dict:
    assert body["success"] is True, body
    data = body["data"]
    for key in (
        "predicted_class",
        "probability",
        "threshold",
        "risk_level",
        "calibrated",
        "warnings",
    ):
        assert key in data, f"Clave faltante '{key}' en respuesta: {data}"

    assert data["predicted_class"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold"] == manifest.threshold
    assert data["risk_level"] in _RISK_LEVELS
    assert isinstance(data["warnings"], list)
    return data


def test_tc07_predict_with_empty_features_imputes_defaults(
    client, predict_url, manifest, empty_features
):
    response = client.post(predict_url, json={"features": empty_features})
    assert response.status_code == 200, response.text
    data = _assert_predict_response_shape(response.json(), manifest)

    assert (data["probability"] >= data["threshold"]) == bool(data["predicted_class"])


def test_tc08_predict_low_risk_patient_returns_class_zero(
    client, predict_url, manifest, low_risk_features
):
    response = client.post(predict_url, json={"features": low_risk_features})
    assert response.status_code == 200, response.text
    data = _assert_predict_response_shape(response.json(), manifest)

    assert data["predicted_class"] == 0, (
        f"Paciente de bajo riesgo clasificado como positivo: prob={data['probability']:.4f}"
    )
    assert data["probability"] < data["threshold"]
    assert data["risk_level"] in {"low", "moderate"}


def test_tc09_predict_high_risk_patient_returns_class_one(
    client, predict_url, manifest, high_risk_features
):
    response = client.post(predict_url, json={"features": high_risk_features})
    assert response.status_code == 200, response.text
    data = _assert_predict_response_shape(response.json(), manifest)

    assert data["predicted_class"] == 1, (
        f"Paciente de alto riesgo clasificado como negativo: prob={data['probability']:.4f}"
    )
    assert data["probability"] >= data["threshold"]
    assert data["risk_level"] in {"elevated", "high"}


def test_tc10_predict_response_contract_metadata_present(
    client, predict_url, manifest, low_risk_features
):
    response = client.post(predict_url, json={"features": low_risk_features})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["success"] is True
    assert body["errors"] is None
    meta = body["meta"]
    assert meta["model_id"] == manifest.model_id
    assert isinstance(meta["request_id"], str) and len(meta["request_id"]) > 0
