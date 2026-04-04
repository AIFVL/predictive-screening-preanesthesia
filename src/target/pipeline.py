from __future__ import annotations

import pandas as pd

from src.target.builder import build_target
from src.target.specific.cancelacion import validate_based_on_cancelacion
from src.target.specific.problemas import validate_based_on_complicaciones_medicas
from src.target.specific.desenlace import validate_based_on_desenlace
from src.target.specific.estancia import validate_based_on_estancia
from src.target.specific.fisiologicas import validate_based_on_fisiologicas
from src.target.specific.induccion import validate_based_on_induccion
from src.target.specific.liquidos import validate_based_on_liquidos
from src.target.specific.reservas import validate_based_on_reservas
from src.target.specific.seguimiento import validate_based_on_seguimiento
from src.target.specific.tecnica import validate_based_on_tecnica
from src.target.specific.tiempos import validate_based_on_tiempos
from src.target.specific.ventilacion import validate_based_on_ventilacion
from src.target.specific.via_aerea import validate_based_on_via_aerea
from src.utils.logger import get_logger

logger = get_logger("target.pipeline")


def apply_all_validations(df: pd.DataFrame, umbral_estancia: int = 3) -> pd.DataFrame:
    """Aplica todos los validadores de flags al DataFrame posoperatorio.

    Each validator is only invoked when its target flag column is not yet
    present in the DataFrame, allowing callers to pass pre-computed flag
    columns directly (e.g. in tests or downstream pipeline stages).
    """
    df = df.copy()

    _FLAG_VALIDATORS = [
        ("flag_cancelacion", lambda d: validate_based_on_cancelacion(d)),
        ("flag_reservas", lambda d: validate_based_on_reservas(d)),
        ("flag_fisiologicas", lambda d: validate_based_on_fisiologicas(d)),
        ("flag_tiempos", lambda d: validate_based_on_tiempos(d)),
        ("flag_induccion", lambda d: validate_based_on_induccion(d)),
        ("flag_via_aerea", lambda d: validate_based_on_via_aerea(d)),
        ("flag_ventilacion", lambda d: validate_based_on_ventilacion(d)),
        ("flag_tecnica", lambda d: validate_based_on_tecnica(d)),
        ("flag_liquidos", lambda d: validate_based_on_liquidos(d)),
        ("flag_desenlace", lambda d: validate_based_on_desenlace(d)),
        ("flag_complicaciones_medicas", lambda d: validate_based_on_complicaciones_medicas(d)),
        ("flag_estancia", lambda d: validate_based_on_estancia(d, umbral_estancia=umbral_estancia)),
        ("flag_seguimiento", lambda d: validate_based_on_seguimiento(d)),
    ]

    for flag_col, validator in _FLAG_VALIDATORS:
        if flag_col not in df.columns:
            df = validator(df)

    return df


def run_target_extraction(
    df_posop: pd.DataFrame,
    target_cfg: dict,
) -> pd.DataFrame:
    """
    Extrae la variable target para una versión específica.

    Args:
        df_posop: DataFrame posoperatorio crudo (o con flags ya calculados).
        target_cfg: dict con threshold, apply_cancel_non_medico_rule, flags.
                    Viene de PipelineConfig.get_target(target_name).

    Returns:
        DataFrame con columna 'target' añadida.
    """
    threshold = int(target_cfg.get("threshold", 1))
    flags = list(target_cfg.get("flags", []))
    apply_cancel_rule = bool(target_cfg.get("apply_cancel_non_medico_rule", True))

    logger.info(f"Extrayendo target: threshold={threshold}, flags={len(flags)}")

    df_validated = apply_all_validations(df_posop)

    df_result = build_target(
        df_validated,
        threshold=threshold,
        flags_to_use=flags,
        excluded_flags=[],
        apply_cancel_non_medico_rule=apply_cancel_rule,
        version_name="",
        verbose=False,
    )

    prevalence = df_result["target"].mean() * 100
    logger.info(
        f"Target extraído: {int(df_result['target'].sum())} positivos "
        f"({prevalence:.1f}%) de {len(df_result)} filas"
    )

    return df_result
