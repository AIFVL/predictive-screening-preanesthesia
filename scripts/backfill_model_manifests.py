from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.cleaning.schema import raw_input_schema_as_dict
from src.models.manifest import build_manifest, manifest_path_for, write_manifest
from src.models.registry import (
    CALIBRATION_FALLBACK_ORDER,
    _resolve_calibration_method,
)
from src.utils.io import read_joblib, read_json, read_parquet, write_joblib
from src.utils.logger import get_logger

logger = get_logger("scripts.backfill_model_manifests")


EXPECTED_ALGORITHMS = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
    "mlp",
    "stacking",
    "voting",
]


def _is_calibrated(model) -> bool:
    """True si el objeto persistido es un CalibratedClassifierCV (ya envuelto)."""
    return isinstance(model, CalibratedClassifierCV)


def _detect_calibration_method(model: CalibratedClassifierCV) -> str | None:
    """Extrae el método ('isotonic' o 'sigmoid') de un CalibratedClassifierCV ya entrenado."""
    method = getattr(model, "method", None)
    if method in ("isotonic", "sigmoid"):
        return method
    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        first = calibrated_classifiers[0]
        for attr in ("calibrator", "calibrators_"):
            cal = getattr(first, attr, None)
            if cal is None:
                continue
            cal_obj = cal[0] if isinstance(cal, list) and cal else cal
            cls_name = type(cal_obj).__name__
            if cls_name == "_SigmoidCalibration":
                return "sigmoid"
            if cls_name == "IsotonicRegression":
                return "isotonic"
    return None


def _algorithm_class_name(algorithm: str) -> str:
    """Mapeo algoritmo → nombre de clase, replicando models_config.yaml."""
    return {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForestClassifier",
        "extra_trees": "ExtraTreesClassifier",
        "xgboost": "XGBClassifier",
        "lightgbm": "LGBMClassifier",
        "hist_gradient_boosting": "HistGradientBoostingClassifier",
        "mlp": "MLPClassifier",
        "stacking": "StackingClassifier",
        "voting": "VotingClassifier",
    }.get(algorithm, "Unknown")


def _to_numeric_filled(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(-1)


def _build_prefit_wrapper(base_model, method: str) -> tuple[CalibratedClassifierCV, str]:
    """
    Construye un CalibratedClassifierCV configurado para "no re-entrenar" el modelo base.

    sklearn ≥1.6 introdujo `FrozenEstimator` (sklearn.frozen) y deprecó `cv="prefit"`;
    sklearn 1.7+ lo removió completamente. Estrategia:
      1. Intentar `FrozenEstimator(base_model)` + cv=5 (API moderna).
      2. Si FrozenEstimator no está disponible (sklearn <1.6), usar `cv="prefit"` (API legacy).

    Retorna (wrapper, cv_label) — `cv_label` es "frozen" o "prefit" para el manifest.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        frozen = FrozenEstimator(base_model)
        return CalibratedClassifierCV(frozen, method=method, cv=5), "frozen"
    except ImportError:
        return CalibratedClassifierCV(base_model, method=method, cv="prefit"), "prefit"


def _try_recalibrate_prefit(
    base_model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    primary_method: str,
) -> tuple[object, dict]:
    methods_to_try = [primary_method] + [m for m in CALIBRATION_FALLBACK_ORDER if m != primary_method]
    last_error: Exception | None = None
    warnings_accum: list[str] = []

    for attempt_method in methods_to_try:
        try:
            wrapper, cv_label = _build_prefit_wrapper(base_model, attempt_method)
            wrapper.fit(X_calib, y_calib)
            fallback_reason = None
            if attempt_method != primary_method:
                fallback_reason = (
                    f"primary_method_{primary_method}_failed: "
                    f"{type(last_error).__name__}: {last_error}"
                    if last_error is not None
                    else f"primary_method_{primary_method}_skipped"
                )
                warnings_accum.append(f"calibration_fallback_to_{attempt_method}")
            warnings_accum.append("calibrated_in_backfill_using_test_split")
            return wrapper, {
                "calibrated": True,
                "method": attempt_method,
                "cv": cv_label,
                "fallback_reason": fallback_reason,
                "warnings": warnings_accum,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Backfill calibration with '{attempt_method}' failed: "
                f"{type(exc).__name__}: {exc}"
            )

    warnings_accum.append("model_not_calibrated")
    return base_model, {
        "calibrated": False,
        "method": None,
        "cv": None,
        "fallback_reason": (
            f"all_methods_failed_in_backfill: {type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "all_methods_failed_in_backfill"
        ),
        "warnings": warnings_accum,
    }


def _process_one_model(
    *,
    target_dir: Path,
    splits_dir: Path,
    target_version: str,
    algorithm: str,
    recalibrate: bool,
) -> dict | None:
    model_path = target_dir / f"{algorithm}_model.joblib"
    metrics_path = target_dir / f"{algorithm}_metrics.json"

    if not model_path.exists():
        logger.info(f"[{target_version}/{algorithm}] sin .joblib — saltando.")
        return None
    if not metrics_path.exists():
        logger.warning(f"[{target_version}/{algorithm}] sin metrics.json — saltando.")
        return None

    logger.info(f"[{target_version}/{algorithm}] cargando modelo...")
    model = read_joblib(model_path)
    metrics_blob = read_json(metrics_path)

    threshold = float(metrics_blob.get("threshold", metrics_blob.get("Threshold", 0.5)))

    # Determinar estado de calibración.
    if _is_calibrated(model):
        method = _detect_calibration_method(model) or "unknown"
        calibration_info = {
            "calibrated": True,
            "method": method,
            "cv": int(getattr(model, "cv", 5)) if isinstance(getattr(model, "cv", 5), int) else None,
            "fallback_reason": None,
            "warnings": [],
        }
        logger.info(f"[{target_version}/{algorithm}] ya está calibrado ({method}).")
    else:
        if recalibrate:
            X_train_path = splits_dir / "X_train.parquet"
            X_test_path = splits_dir / "X_test.parquet"
            y_test_path = splits_dir / "y_test.parquet"
            if not (X_test_path.exists() and y_test_path.exists()):
                logger.warning(
                    f"[{target_version}/{algorithm}] no se puede recalibrar: faltan X_test/y_test."
                )
                calibration_info = {
                    "calibrated": False,
                    "method": None,
                    "cv": None,
                    "fallback_reason": "missing_test_split_for_prefit_calibration",
                    "warnings": ["model_not_calibrated"],
                }
            else:
                X_test = _to_numeric_filled(read_parquet(X_test_path))
                y_test = read_parquet(y_test_path)["target"]
                primary_method = _resolve_calibration_method(_algorithm_class_name(algorithm))
                logger.info(
                    f"[{target_version}/{algorithm}] recalibrando con cv='prefit' "
                    f"(método primario={primary_method})..."
                )
                new_model, calibration_info = _try_recalibrate_prefit(
                    model, X_test, y_test, primary_method=primary_method
                )
                if calibration_info["calibrated"]:
                    backup_path = target_dir / f"{algorithm}_model.uncalibrated.bak.joblib"
                    if not backup_path.exists():
                        shutil.copy2(model_path, backup_path)
                        logger.info(f"  backup → {backup_path.name}")
                    write_joblib(new_model, model_path)
                    logger.info(f"  modelo recalibrado guardado en {model_path.name}")
                    model = new_model
        else:
            calibration_info = {
                "calibrated": False,
                "method": None,
                "cv": None,
                "fallback_reason": "model_artifact_was_not_calibrated_at_train_time",
                "warnings": ["model_not_calibrated"],
            }
            logger.info(
                f"[{target_version}/{algorithm}] no calibrado — manifest reflejará "
                f"calibrated=false (usar --recalibrate para intentar arreglarlo)."
            )

    X_train_path = splits_dir / "X_train.parquet"
    y_train_path = splits_dir / "y_train.parquet"
    if not (X_train_path.exists() and y_train_path.exists()):
        logger.error(
            f"[{target_version}/{algorithm}] faltan X_train/y_train en {splits_dir} — "
            f"no se puede construir manifest."
        )
        return None
    X_train = read_parquet(X_train_path)
    y_train = read_parquet(y_train_path)["target"]

    metrics_for_manifest = {
        "ROC_AUC": metrics_blob.get("ROC_AUC"),
        "PR_AUC": metrics_blob.get("PR_AUC"),
        "F2": metrics_blob.get("F2"),
        "F1": metrics_blob.get("F1"),
        "Recall": metrics_blob.get("Recall"),
        "Precision": metrics_blob.get("Precision"),
        "Specificity": metrics_blob.get("Specificity"),
        "Balanced_Accuracy": metrics_blob.get("Balanced_Accuracy"),
        "Brier": metrics_blob.get("Brier"),
        "Accuracy": metrics_blob.get("Accuracy"),
    }

    manifest = build_manifest(
        target_version=target_version,
        algorithm=algorithm,
        model_filename=model_path.name,
        metrics_filename=metrics_path.name,
        X_train=X_train,
        y_train=y_train,
        threshold=threshold,
        threshold_metric="f2",
        calibration_info=calibration_info,
        metrics=metrics_for_manifest,
        extra_warnings=["generated_by_backfill"],
        raw_input_schema=raw_input_schema_as_dict(),
    )
    out_manifest = manifest_path_for(target_dir, algorithm)
    write_manifest(manifest, out_manifest)

    return {
        "target_version": target_version,
        "algorithm": algorithm,
        "manifest_path": str(out_manifest),
        "calibrated": calibration_info["calibrated"],
        "method": calibration_info["method"],
    }


def run(
    root: Path,
    targets: list[str],
    recalibrate: bool,
) -> list[dict]:
    """Procesa todos los targets y emite un resumen."""
    results: list[dict] = []
    for target_version in targets:
        target_dir = root / "models" / target_version
        splits_dir = root / "data_processed" / target_version / "splits"
        if not target_dir.is_dir():
            logger.warning(f"No existe {target_dir} — saltando.")
            continue
        if not splits_dir.is_dir():
            logger.warning(f"No existe {splits_dir} — saltando.")
            continue

        logger.info(f"== Procesando target_version={target_version} ==")
        for algorithm in EXPECTED_ALGORITHMS:
            res = _process_one_model(
                target_dir=target_dir,
                splits_dir=splits_dir,
                target_version=target_version,
                algorithm=algorithm,
                recalibrate=recalibrate,
            )
            if res is not None:
                results.append(res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/v2"),
        help="Directorio raíz que contiene `models/<target>/` y `data_processed/<target>/splits/`. Default: output/v2",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Lista de target_version a procesar (ej: target_d_v2_hosp target_f_predictibilidad_maxima)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Si se pasa, recalibra modelos no calibrados sobre X_test/y_test (cv='prefit') y reemplaza el .joblib (con backup).",
    )
    args = parser.parse_args()

    results = run(root=args.root, targets=args.targets, recalibrate=args.recalibrate)

    print("\n=== Resumen del backfill ===")
    print(f"Total manifests generados: {len(results)}")
    calibrated = sum(1 for r in results if r["calibrated"])
    uncalibrated = len(results) - calibrated
    print(f"  calibrados:    {calibrated}")
    print(f"  no calibrados: {uncalibrated}")
    if uncalibrated:
        print("Modelos sin calibrar (servidos con `warnings: model_not_calibrated`):")
        for r in results:
            if not r["calibrated"]:
                print(f"  - {r['target_version']}/{r['algorithm']}")


if __name__ == "__main__":
    main()
