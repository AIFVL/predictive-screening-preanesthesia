from __future__ import annotations

import time

import pytest

SINGLE_LATENCY_MS_LIMIT = 1500.0
BATCH_LATENCY_MS_LIMIT = 5000.0
BATCH_SIZE = 50
WARMUP_REPS = 1
MEASURE_REPS = 5


def _post_and_time_ms(client, url: str, payload: dict) -> float:
    start = time.perf_counter()
    response = client.post(url, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True, response.text
    return elapsed_ms


def test_tc20_predict_single_latency_under_threshold(
    client, predict_url, low_risk_features
):
    payload = {"features": low_risk_features}
    for _ in range(WARMUP_REPS):
        _post_and_time_ms(client, predict_url, payload)

    samples = [_post_and_time_ms(client, predict_url, payload) for _ in range(MEASURE_REPS)]
    samples.sort()
    median = samples[len(samples) // 2]
    assert median < SINGLE_LATENCY_MS_LIMIT, (
        f"Latencia mediana {median:.1f} ms ≥ umbral {SINGLE_LATENCY_MS_LIMIT:.0f} ms. "
        f"Muestras: {[round(s, 1) for s in samples]}"
    )


def test_tc21_predict_batch_latency_scales_subquadratically(
    client, batch_predict_url, low_risk_features
):
    items = [low_risk_features for _ in range(BATCH_SIZE)]
    payload = {"items": items}

    _post_and_time_ms(client, batch_predict_url, payload)

    elapsed = _post_and_time_ms(client, batch_predict_url, payload)
    if elapsed >= BATCH_LATENCY_MS_LIMIT:
        pytest.fail(
            f"Latencia batch {elapsed:.1f} ms ≥ umbral {BATCH_LATENCY_MS_LIMIT:.0f} ms "
            f"para N={BATCH_SIZE}."
        )
