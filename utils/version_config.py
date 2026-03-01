from __future__ import annotations

from datetime import date


VERSION_HISTORY: list[dict] = [
    {
        "version": "v1",
        "threshold": 1,
        "flags_to_use": "RELEVANT_FLAGS",
        "excluded_flags": "EXCLUDED_FLAGS",
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v1.xlsx",
        "description": "Sin agregación (>=1 flag)",
        "executed_on": "2026-02-28",
        "notes": "Baseline histórico",
    },
    {
        "version": "v2",
        "threshold": 2,
        "flags_to_use": "RELEVANT_FLAGS",
        "excluded_flags": "EXCLUDED_FLAGS",
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v2.xlsx",
        "description": "Con agregación (>=2 flags)",
        "executed_on": "2026-02-28",
        "notes": "Baseline histórico",
    },
    {
        "version": "v3_sin_liquidos",
        "threshold": 1,
        "flags_to_use": "RELEVANT_FLAGS - flag_liquidos",
        "excluded_flags": "EXCLUDED_FLAGS",
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v3_sin_liquidos.xlsx",
        "description": "Sin flag_liquidos (>=1 flag)",
        "executed_on": "2026-02-28",
        "notes": "Sensibilidad clínica",
    },
]


# Versiones activas para TODO el flujo multi-versión (target -> merge -> selección -> modelado)
ACTIVE_VERSIONS: list[str] = [
    "v1",
    "v2",
    "v3_sin_liquidos",
]


def build_target_versions_config(relevant_flags: list[str], excluded_flags: list[str]) -> list[dict]:
    """
    Construye configuración de versiones para target extraction usando ACTIVE_VERSIONS.
    """
    catalog = {
        "v1": {
            "name": "v1",
            "threshold": 1,
            "flags_to_use": relevant_flags,
            "excluded_flags": excluded_flags,
            "apply_cancel_non_medico_rule": True,
            "export_name": "OPERA_POS_v1.xlsx",
            "description": "Sin agregación (>=1 flag)",
        },
        "v2": {
            "name": "v2",
            "threshold": 2,
            "flags_to_use": relevant_flags,
            "excluded_flags": excluded_flags,
            "apply_cancel_non_medico_rule": True,
            "export_name": "OPERA_POS_v2.xlsx",
            "description": "Con agregación (>=2 flags)",
        },
        "v3_sin_liquidos": {
            "name": "v3_sin_liquidos",
            "threshold": 1,
            "flags_to_use": [flag for flag in relevant_flags if flag != "flag_liquidos"],
            "excluded_flags": excluded_flags,
            "apply_cancel_non_medico_rule": True,
            "export_name": "OPERA_POS_v3_sin_liquidos.xlsx",
            "description": "Sin flag_liquidos (>=1 flag)",
        },
    }

    missing = [version for version in ACTIVE_VERSIONS if version not in catalog]
    if missing:
        raise ValueError(f"Versiones activas sin definición en catálogo: {missing}")

    return [catalog[version] for version in ACTIVE_VERSIONS]


def get_active_target_export_names() -> list[str]:
    """
    Lista de archivos OPERA_POS_*.xlsx esperados para ACTIVE_VERSIONS.
    """
    export_by_version = {
        "v1": "OPERA_POS_v1.xlsx",
        "v2": "OPERA_POS_v2.xlsx",
        "v3_sin_liquidos": "OPERA_POS_v3_sin_liquidos.xlsx",
    }

    missing = [version for version in ACTIVE_VERSIONS if version not in export_by_version]
    if missing:
        raise ValueError(f"Versiones activas sin nombre de export definido: {missing}")

    return [export_by_version[version] for version in ACTIVE_VERSIONS]


def append_history_entry(version: str, notes: str = "") -> dict:
    """
    Helper opcional para registrar nuevas ejecuciones en memoria.
    """
    entry = {
        "version": version,
        "executed_on": str(date.today()),
        "notes": notes,
    }
    VERSION_HISTORY.append(entry)
    return entry
