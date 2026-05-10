from __future__ import annotations


def test_tc03_targets_endpoint_lists_recommended_target(client, target):
    response = client.get("/targets")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    slugs = {t["slug"] for t in body["data"]}
    assert target in slugs, f"Falta el slug '{target}' en la respuesta: {slugs}"

    entry = next(t for t in body["data"] if t["slug"] == target)
    assert entry["n_models"] >= 1
    assert isinstance(entry["display_name"], str)


def test_tc04_models_endpoint_lists_target_algorithm(client, target, algorithm):
    response = client.get("/models")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True

    pairs = {(m["target"], m["algorithm"]) for m in body["data"]}
    assert (target, algorithm) in pairs, (
        f"Falta {target}/{algorithm} en la lista de modelos: {sorted(pairs)}"
    )


def test_tc05_model_detail_returns_metadata(client, target, algorithm, manifest):
    response = client.get(f"/models/{target}/{algorithm}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    data = body["data"]

    assert data["model_id"] == manifest.model_id
    assert data["target"] == target
    assert data["algorithm"] == algorithm
    assert data["threshold"] == manifest.threshold
    assert data["calibrated"] == manifest.calibrated
    assert data["n_features"] == len(manifest.feature_names)


def test_tc06_model_schema_lists_features(client, target, algorithm, manifest):
    response = client.get(f"/models/{target}/{algorithm}/schema")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    data = body["data"]

    assert data["target"] == target
    assert data["algorithm"] == algorithm
    assert data["threshold"] == manifest.threshold

    schema_names = [f["name"] for f in data["features"]]
    assert schema_names == manifest.feature_names, (
        "El orden y composición de features del schema debe coincidir con el manifest."
    )

    for feature in data["features"]:
        assert feature["dtype"] in {"int64", "float64", "bool"}, feature
        assert feature["required"] is False
