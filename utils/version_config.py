from __future__ import annotations

from datetime import date

from utils.target_extraction.constants import ALL_FLAGS, RELEVANT_FLAGS


DEFAULT_FLAGS_V1 = list(ALL_FLAGS)
DEFAULT_FLAGS_V2 = list(ALL_FLAGS)
DEFAULT_FLAGS_V3_SIN_LIQUIDOS = [flag for flag in RELEVANT_FLAGS if flag != "flag_liquidos"]

PREOP_FOCUS_FLAGS = [
    "flag_cancelacion",
    "flag_reservas",
    "flag_via_aerea",
    "flag_induccion",
    "flag_tiempos",
    "flag_tecnica",
    "flag_complicaciones_medicas",
    "flag_estancia",
    "flag_seguimiento",
]

PREOP_STRICT_FLAGS = [
    "flag_cancelacion",
    "flag_reservas",
    "flag_via_aerea",
    "flag_complicaciones_medicas",
]

# ---------------------------------------------------------------------------
# 3 versiones clínicas congeladas (Target A / B / C)
# ---------------------------------------------------------------------------
# Target A – Sensible: todas las flags, threshold=1
TARGET_A_FLAGS = list(ALL_FLAGS)
# Target B – Clínicamente Relevante: flags preop estrictas, threshold=1
TARGET_B_FLAGS = list(PREOP_FOCUS_FLAGS)
# Target C – Alta Severidad: mismas flags que B, threshold=2
TARGET_C_FLAGS = list(PREOP_FOCUS_FLAGS)


RECOMMENDED_EXPERIMENTAL_VERSION_SPECS: list[dict] = [
    {
        "version": "target_a_sensible",
        "threshold": 1,
        "flags_to_use": TARGET_A_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_a_sensible.xlsx",
        "description": "Target A - Sensible: flags relevantes, threshold=1",
    },
    {
        "version": "target_b_clinicamente_relevante",
        "threshold": 1,
        "flags_to_use": TARGET_B_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_b_clinicamente_relevante.xlsx",
        "description": "Target B - Clínicamente Relevante: flags preop estrictas, threshold=1",
    },
    {
        "version": "target_c_alta_severidad",
        "threshold": 2,
        "flags_to_use": TARGET_C_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_c_alta_severidad.xlsx",
        "description": "Target C - Alta Severidad: mismas flags que B, threshold=2",
    },
]


TARGET_VERSION_CATALOG: dict[str, dict] = {
    "target_a_sensible": {
        "name": "target_a_sensible",
        "threshold": 1,
        "flags_to_use": TARGET_A_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_a_sensible.xlsx",
        "description": "Target A - Sensible: flags relevantes, threshold=1",
    },
    "target_b_clinicamente_relevante": {
        "name": "target_b_clinicamente_relevante",
        "threshold": 1,
        "flags_to_use": TARGET_B_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_b_clinicamente_relevante.xlsx",
        "description": "Target B - Clínicamente Relevante: flags preop estrictas, threshold=1",
    },
    "target_c_alta_severidad": {
        "name": "target_c_alta_severidad",
        "threshold": 2,
        "flags_to_use": TARGET_C_FLAGS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_target_c_alta_severidad.xlsx",
        "description": "Target C - Alta Severidad: mismas flags que B, threshold=2",
    },
}


VERSION_HISTORY: list[dict] = [
    {
        "version": "v1",
        "threshold": 1,
        "flags_to_use": DEFAULT_FLAGS_V1,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v1.xlsx",
        "description": "Baseline completo con TODAS las flags (>=1 flag)",
        "executed_on": "2026-02-28",
        "notes": "Baseline histórico completo",
    },
    {
        "version": "v2",
        "threshold": 2,
        "flags_to_use": DEFAULT_FLAGS_V2,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v2.xlsx",
        "description": "Con agregación (>=2 flags)",
        "executed_on": "2026-02-28",
        "notes": "Baseline histórico",
    },
    {
        "version": "v3_sin_liquidos",
        "threshold": 1,
        "flags_to_use": DEFAULT_FLAGS_V3_SIN_LIQUIDOS,
        "apply_cancel_non_medico_rule": True,
        "export_name": "OPERA_POS_v3_sin_liquidos.xlsx",
        "description": "Sin flag_liquidos (>=1 flag)",
        "executed_on": "2026-02-28",
        "notes": "Sensibilidad clínica",
    },
]


# Versiones activas para TODO el flujo multi-versión (target -> merge -> selección -> modelado)
ACTIVE_VERSIONS: list[str] = [
    "target_a_sensible",
    "target_b_clinicamente_relevante",
    "target_c_alta_severidad",
]


def build_target_versions_config(
    version_specs: list[dict] | None = None,
    relevant_flags: list[str] | None = None,
    excluded_flags: list[str] | None = None,
) -> list[dict]:
    """
    Construye configuración de versiones para target extraction.

    Si `version_specs` es None, usa `ACTIVE_VERSIONS` sobre `TARGET_VERSION_CATALOG`.
    Si se envía `version_specs`, cada elemento debe traer explícitamente:
      - version/name
      - flags_to_use
    y opcionalmente threshold, apply_cancel_non_medico_rule, export_name, description.

    Compatibilidad: `relevant_flags`/`excluded_flags` se aceptan para no romper
    llamadas antiguas, pero el flujo recomendado es versión explícita con flags_to_use.
    """
    if relevant_flags is not None and version_specs is None:
        overridden_catalog = {
            key: dict(value)
            for key, value in TARGET_VERSION_CATALOG.items()
        }
        for key, value in overridden_catalog.items():
            if "sin_liquidos" in key:
                value["flags_to_use"] = [
                    flag for flag in relevant_flags if flag != "flag_liquidos"
                ]
            else:
                value["flags_to_use"] = list(relevant_flags)

        missing = [version for version in ACTIVE_VERSIONS if version not in overridden_catalog]
        if missing:
            raise ValueError(f"Versiones activas sin definición en catálogo: {missing}")
        return [dict(overridden_catalog[version]) for version in ACTIVE_VERSIONS]

    if version_specs is None:
        missing = [version for version in ACTIVE_VERSIONS if version not in TARGET_VERSION_CATALOG]
        if missing:
            raise ValueError(f"Versiones activas sin definición en catálogo: {missing}")
        return [dict(TARGET_VERSION_CATALOG[version]) for version in ACTIVE_VERSIONS]

    if not version_specs:
        raise ValueError("Debes definir al menos una versión en version_specs")

    normalized_versions = []
    seen = set()
    for index, version in enumerate(version_specs, start=1):
        version_name = str(version.get("version") or version.get("name") or "").strip()
        if not version_name:
            raise ValueError(f"La versión #{index} no tiene nombre")
        if version_name in seen:
            raise ValueError(f"Nombre de versión duplicado: {version_name}")
        seen.add(version_name)

        flags_to_use = version.get("flags_to_use")
        if not flags_to_use:
            raise ValueError(f"La versión '{version_name}' debe definir flags_to_use")

        normalized_versions.append(
            {
                "name": version_name,
                "threshold": int(version.get("threshold", 1)),
                "flags_to_use": list(flags_to_use),
                "apply_cancel_non_medico_rule": bool(
                    version.get("apply_cancel_non_medico_rule", True)
                ),
                "export_name": version.get("export_name", f"OPERA_POS_{version_name}.xlsx"),
                "description": version.get("description", ""),
            }
        )

    return normalized_versions


def get_active_target_export_names() -> list[str]:
    """
    Lista de archivos OPERA_POS_*.xlsx esperados para ACTIVE_VERSIONS.
    """
    export_by_version = {
        version: config["export_name"]
        for version, config in TARGET_VERSION_CATALOG.items()
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


def get_recommended_experimental_versions() -> list[dict]:
    """
    Devuelve el set recomendado de versiones (3 versiones clínicas congeladas).
    """
    return build_target_versions_config(version_specs=RECOMMENDED_EXPERIMENTAL_VERSION_SPECS)


def get_target_summary_table() -> "pd.DataFrame":
    """
    Tabla resumen de las versiones activas:
    Version | Threshold | Flags_Used | N_Flags | Description
    (N_Positivos y Prevalencia se calculan con datos reales en el pipeline).
    """
    import pandas as pd
    rows = []
    for v in ACTIVE_VERSIONS:
        spec = TARGET_VERSION_CATALOG[v]
        rows.append({
            "Version": v,
            "Threshold": spec["threshold"],
            "Flags_Used": ", ".join(spec["flags_to_use"]),
            "N_Flags": len(spec["flags_to_use"]),
            "Description": spec["description"],
        })
    return pd.DataFrame(rows)
