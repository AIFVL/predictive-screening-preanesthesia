from pathlib import Path

import pandas as pd

from .constants import EXCLUDED_FLAGS, RELEVANT_FLAGS
from .target_builder import build_target
from .specific.cancelacion import validate_based_on_cancelacion 
from .specific.problemas import validate_based_on_complicaciones_medicas
from .specific.desenlace import validate_based_on_desenlace
from .specific.estancia import validate_based_on_estancia
from .specific.fisiologicas import validate_based_on_fisiologicas
from .specific.induccion import validate_based_on_induccion
from .specific.liquidos import validate_based_on_liquidos
from .specific.reservas import validate_based_on_reservas
from .specific.seguimiento import validate_based_on_seguimiento
from .specific.tecnica import validate_based_on_tecnica
from .specific.tiempos import validate_based_on_tiempos
from .specific.ventilacion import validate_based_on_ventilacion
from .specific.via_aerea import validate_based_on_via_aerea

def apply_all_validations(df: pd.DataFrame, umbral_estancia: int = 3) -> pd.DataFrame:
    df_result = df.copy()
    df_result = validate_based_on_cancelacion(df_result)
    df_result = validate_based_on_reservas(df_result)
    df_result = validate_based_on_fisiologicas(df_result)
    df_result = validate_based_on_tiempos(df_result)
    df_result = validate_based_on_induccion(df_result)
    df_result = validate_based_on_via_aerea(df_result)
    df_result = validate_based_on_ventilacion(df_result)
    df_result = validate_based_on_tecnica(df_result)
    df_result = validate_based_on_liquidos(df_result)
    df_result = validate_based_on_desenlace(df_result)
    df_result = validate_based_on_complicaciones_medicas(df_result)
    df_result = validate_based_on_estancia(df_result, umbral_estancia=umbral_estancia)
    df_result = validate_based_on_seguimiento(df_result)
    return df_result


def _normalize_versions_config(
    versions: list[dict] | None,
    threshold_v1: int,
    threshold_v2: int,
    export_v1_name: str,
    export_v2_name: str,
) -> tuple[list[dict], bool]:
    if versions is None:
        return [
            {
                "name": "v1",
                "threshold": threshold_v1,
                "flags_to_use": RELEVANT_FLAGS,
                "excluded_flags": EXCLUDED_FLAGS,
                "apply_cancel_non_medico_rule": True,
                "export_name": export_v1_name,
                "description": "Sin agregación (≥1 flag)",
            },
            {
                "name": "v2",
                "threshold": threshold_v2,
                "flags_to_use": RELEVANT_FLAGS,
                "excluded_flags": EXCLUDED_FLAGS,
                "apply_cancel_non_medico_rule": True,
                "export_name": export_v2_name,
                "description": "Con agregación (≥2 flags)",
            },
        ], True

    if not versions:
        raise ValueError("Debes definir al menos una versión en 'versions'.")

    normalized = []
    seen_names = set()
    for i, version in enumerate(versions, start=1):
        name = str(version.get("name", f"v{i}")).strip()
        if not name:
            raise ValueError(f"La versión #{i} tiene un nombre vacío.")
        if name in seen_names:
            raise ValueError(f"Nombre de versión duplicado: '{name}'.")
        seen_names.add(name)

        flags_to_use = version.get("flags_to_use")
        if flags_to_use is None:
            flags_to_use = version.get("relevant_flags")
        if flags_to_use is None:
            raise ValueError(
                f"La versión '{name}' debe definir 'flags_to_use' con la lista de flags a usar."
            )

        normalized.append(
            {
                "name": name,
                "threshold": int(version.get("threshold", 1)),
                "flags_to_use": flags_to_use,
                "excluded_flags": version.get("excluded_flags", EXCLUDED_FLAGS),
                "apply_cancel_non_medico_rule": bool(
                    version.get("apply_cancel_non_medico_rule", True)
                ),
                "export_name": version.get("export_name", f"OPERA_POS_{name}.xlsx"),
                "description": version.get("description", ""),
            }
        )

    return normalized, False


def run_target_extraction_pipeline(
    df_postqx: pd.DataFrame,
    threshold_v1: int = 1,
    threshold_v2: int = 2,
    versions: list[dict] | None = None,
    umbral_estancia: int = 3,
    export_dir: str = "../data",
    export_v1_name: str = "OPERA_POS_v1.xlsx",
    export_v2_name: str = "OPERA_POS_v2.xlsx",
    export: bool = True,
):
    print("=" * 70)
    print("INICIANDO PIPELINE DE EXTRACCIÓN")
    print("=" * 70)

    df_validated = apply_all_validations(df_postqx.copy(), umbral_estancia=umbral_estancia)
    normalized_versions, legacy_mode = _normalize_versions_config(
        versions=versions,
        threshold_v1=threshold_v1,
        threshold_v2=threshold_v2,
        export_v1_name=export_v1_name,
        export_v2_name=export_v2_name,
    )

    df_versions = {}
    print(f"\n{'=' * 70}")
    print(f"CONSTRUCCIÓN DE VERSIONES ({len(normalized_versions)})")
    print(f"{'=' * 70}")

    for version in normalized_versions:
        version_name = version["name"]
        version_description = version.get("description", "")
        if version_description:
            print(f"\n--- {version_name}: {version_description}")
        else:
            print(f"\n--- {version_name}")

        df_versions[version_name] = build_target(
            df_validated,
            threshold=version["threshold"],
            flags_to_use=version["flags_to_use"],
            excluded_flags=version["excluded_flags"],
            apply_cancel_non_medico_rule=version["apply_cancel_non_medico_rule"],
            version_name=version_name,
        )

    print(f"\n{'=' * 70}")
    print("COMPARACIÓN DE VERSIONES DEL TARGET")
    print(f"{'=' * 70}")
    print(f"\n{'Versión':<18} {'Threshold':<10} {'Positivos':<12} {'Prevalencia':<12}")
    print("-" * 60)
    for version in normalized_versions:
        version_name = version["name"]
        df_version = df_versions[version_name]
        positives = int(df_version["target"].sum())
        prevalence = df_version["target"].mean() * 100
        print(
            f"{version_name:<18} {version['threshold']:<10} "
            f"{positives:<12,} {prevalence:<11.2f}%"
        )

    if export:
        export_base = Path(export_dir)
        for version in normalized_versions:
            version_name = version["name"]
            export_name = version["export_name"]
            export_path = export_base / export_name
            df_version = df_versions[version_name]
            df_version.to_excel(export_path, index=False)

            print(f"\n✓ Dataset {version_name} exportado: {export_path}")
            print(f"  → {df_version.shape[0]:,} filas × {df_version.shape[1]} columnas")
            print(
                f"  → Target: {df_version['target'].sum():,} positivos "
                f"({df_version['target'].mean() * 100:.2f}%)"
            )

    if legacy_mode:
        return df_validated, df_versions["v1"], df_versions["v2"]

    return df_validated, df_versions
