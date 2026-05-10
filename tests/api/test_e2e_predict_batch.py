from __future__ import annotations

import os


def _max_batch_size() -> int:
    return int(os.environ.get("MAX_BATCH_SIZE", "100"))


def test_tc16_batch_predict_happy_path_returns_n_predictions(
    client, batch_predict_url, low_risk_features, high_risk_features, manifest
):
    items = [
        low_risk_features,
        high_risk_features,
        {},
    ]
    response = client.post(batch_predict_url, json={"items": items})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    data = body["data"]

    assert data["n"] == len(items)
    assert len(data["predictions"]) == len(items)
    for pred in data["predictions"]:
        assert pred["predicted_class"] in (0, 1)
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["threshold"] == manifest.threshold


def test_tc17_batch_predict_empty_returns_empty_batch_error(client, batch_predict_url):
    response = client.post(batch_predict_url, json={"items": []})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False
    assert body["errors"][0]["code"] == "empty_batch"


def test_tc18_batch_predict_oversized_returns_batch_too_large(
    client, batch_predict_url, low_risk_features
):
    n = _max_batch_size() + 1
    items = [low_risk_features for _ in range(n)]
    response = client.post(batch_predict_url, json={"items": items})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False
    err = body["errors"][0]
    assert err["code"] == "batch_too_large"
    assert str(_max_batch_size()) in err["message"]


def test_tc19_batch_predict_with_invalid_item_is_rejected(
    client, batch_predict_url, low_risk_features
):
    bad_item = {**low_risk_features, "feature_que_no_existe": 1}
    response = client.post(batch_predict_url, json={"items": [low_risk_features, bad_item]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False
    assert body["errors"][0]["code"] == "invalid_input"
