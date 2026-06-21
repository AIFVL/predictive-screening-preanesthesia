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
def predict_url(target, algorithm) -> str:
    return f"/models/{target}/{algorithm}/predict"


@pytest.fixture
def low_risk_patient() -> dict:
    """Paciente de bajo riesgo clínico — datos crudos con claves de columna del dataset."""
    return {
        "Edad": 35,
        "Sexo": "Masculino",
        "Atención": "Electivo",
        "Peso (Kg)": 68,
        "Talla (cm)": 170,
        "IMC": 23.5,
        "Tipo de anestesia propuesta": "Local",
        "Examen_Hemoglobina(g/dl)": 14.2,
        "Dx Preoperatorio": "procedimiento menor electivo sin comorbilidades",
        "Procedimiento propuesto": "biopsia de piel",
    }


@pytest.fixture
def high_risk_patient() -> dict:
    """Paciente de alto riesgo clínico — datos crudos con claves de columna del dataset."""
    return {
        "Edad": 78,
        "Sexo": "Masculino",
        "Atención": "Urgente",
        "Peso (Kg)": 92,
        "Talla (cm)": 165,
        "IMC": 33.8,
        "Tipo de anestesia propuesta": "General",
        "Examen_Hemoglobina(g/dl)": 9.1,
        "Dx Preoperatorio": "insuficiencia cardíaca congestiva descompensada, sepsis",
        "Procedimiento propuesto": "laparotomía exploratoria de emergencia",
        "Antecedentes cardiovasculares": "hipertensión arterial, fibrilación auricular",
        "Antecedente endocrinológicos": "diabetes mellitus tipo 2 insulinorrequiriente",
    }
