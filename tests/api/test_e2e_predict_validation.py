from __future__ import annotations


def _assert_invalid_input(body: dict, expected_field_substring: str | None = None) -> dict:
    assert body["success"] is False
    assert body["data"] is None
    assert body["errors"] and len(body["errors"]) >= 1
    err = body["errors"][0]
    assert err["code"] == "invalid_input", err
    if expected_field_substring is not None:
        assert err.get("field") and expected_field_substring in err["field"], err
    return err


def test_tc11_predict_missing_features_key_returns_invalid_input(client, predict_url):
    response = client.post(predict_url, json={})
    assert response.status_code == 200, response.text
    _assert_invalid_input(response.json(), expected_field_substring="features")


def test_tc12_predict_unknown_feature_is_rejected(client, predict_url):
    payload = {"features": {"feature_inexistente_xyz": 1}}
    response = client.post(predict_url, json=payload)
    assert response.status_code == 200, response.text
    err = _assert_invalid_input(response.json())
    assert "feature_inexistente_xyz" in (err.get("field") or "") or "extra" in err["message"].lower()


def test_tc13_predict_invalid_dtype_is_rejected(client, predict_url, feature_names):
    numeric_feature = next(
        name for name in feature_names
        if name not in ("Sexo_encoded",)  # cualquier numérica del manifest
    )
    payload = {"features": {numeric_feature: "no-soy-numero"}}
    response = client.post(predict_url, json=payload)
    assert response.status_code == 200, response.text
    _assert_invalid_input(response.json(), expected_field_substring=numeric_feature)


def test_tc14_predict_unknown_target_returns_404(client, algorithm):
    response = client.post(
        f"/models/no_existe_target/{algorithm}/predict",
        json={"features": {}},
    )
    assert response.status_code == 404, response.text
    assert "no_existe_target" in response.json().get("detail", "")


def test_tc15_predict_unknown_algorithm_returns_404(client, target):
    response = client.post(
        f"/models/{target}/no_existe_algo/predict",
        json={"features": {}},
    )
    assert response.status_code == 404, response.text
    assert "no_existe_algo" in response.json().get("detail", "")
