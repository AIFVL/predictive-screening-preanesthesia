from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def _configure_env() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("MODELS_DIR", str(PROJECT_ROOT / "output" / "v2" / "models"))
    os.environ.setdefault("MAX_BATCH_SIZE", "100")
    os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def app():
    from api.main import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def target() -> str:
    return "hospitalization_risk"


@pytest.fixture(scope="session")
def algorithm() -> str:
    return "xgboost"


@pytest.fixture(scope="session")
def manifest(client, app, target, algorithm):
    registry = app.state.registry
    m = registry.get_manifest(target, algorithm)
    if m is None:
        pytest.skip(
            f"Modelo {target}/{algorithm} no registrado — "
            "asegúrate de haber entrenado el modelo y de que MODELS_DIR apunte al output correcto."
        )
    return m


@pytest.fixture(scope="session")
def feature_names(manifest) -> list[str]:
    return list(manifest.feature_names)


@pytest.fixture(scope="session")
def threshold(manifest) -> float:
    return float(manifest.threshold)


@pytest.fixture
def empty_features() -> dict:
    return {}


@pytest.fixture
def low_risk_features() -> dict:
    return {
        "Edad": 35,
        "Peso (Kg)": 68,
        "Talla (cm)": 170,
        "IMC": 23.5,
        "Tipo de anestesia propuesta_local": 1,
        "Tipo de anestesia propuesta_general": 0,
        "score_proc_low_severity": 1,
        "score_proc_high_severity": 0,
        "score_proc_critical": 0,
        "score_dx_critical": 0,
        "score_dx_high_severity": 0,
        "Examen_Hemoglobina(g/dl)": 14.2,
        "Antecedente endocrinológicos_negativo": 1,
        "Antecedente renales_negativo": 1,
        "Antecedente hematológicos _negativo": 1,
    }


@pytest.fixture
def high_risk_features() -> dict:
    return {
        "Edad": 78,
        "Peso (Kg)": 92,
        "Talla (cm)": 165,
        "IMC": 33.8,
        "Tipo de anestesia propuesta_general": 1,
        "Tipo de anestesia propuesta_local": 0,
        "score_proc_critical": 1,
        "score_proc_high_severity": 1,
        "score_proc_low_severity": 0,
        "score_dx_critical": 1,
        "score_dx_high_severity": 1,
        "Examen_Hemoglobina(g/dl)": 9.1,
    }


def _predict_url(target: str, algorithm: str) -> str:
    return f"/models/{target}/{algorithm}/predict"


def _batch_predict_url(target: str, algorithm: str) -> str:
    return f"/models/{target}/{algorithm}/predict/batch"


@pytest.fixture
def predict_url(target, algorithm) -> str:
    return _predict_url(target, algorithm)


@pytest.fixture
def batch_predict_url(target, algorithm) -> str:
    return _batch_predict_url(target, algorithm)
