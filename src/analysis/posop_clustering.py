# src/analysis/posop_clustering.py
"""
Enfoque A: ¿Qué estructuras internas existen en el dataset posoperatorio?

Agrupa pacientes según su patrón de flags posoperatorios usando KMeans.
Descubre si existen tipos naturales de complicación y si sus perfiles
preoperatorios difieren entre clusters.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("analysis.posop_clustering")

_FLAGS_CLUSTERING = [
    "flag_intubacion_dificil",
    "flag_tipo_intubacion_complejo",
    "flag_elemento_via_aerea_complejo",
    "flag_aferesis",
    "flag_balance_extremo",
    "flag_liquidos",
    "flag_destino_uci",
    "flag_intubado_salida",
    "flag_complicaciones_pulmonares",
    "flag_complicaciones_medicas",
    "flag_estancia_prolongada",
    "flag_hospitalizacion_no_anticipada",
    "flag_uci_no_planeada",
    "flag_estancia_uci",
    "flag_urgencias_30_dias",
    "flag_interconsultas",
    "flag_reserva_sangre",
    "flag_reserva_hemoderivados",
]

_PREOP_PROFILE_COLS = [
    "Edad", "IMC",
    "Tensión Arterial Sistólica (mm/Hg)",
    "score_proc_high_severity", "score_proc_moderate_severity",
    "score_dx_high_severity",
    "Puntaje Mallampati",
    "Antecedentes cardiovasculares_hta",
    "Antecedente endocrinológicos_diabetes",
    "Antecedente respiratorios_epoc",
    "Antecedente respiratorios_asma",
]


def run_posop_clustering(
    posop_path: str | Path,
    merged_path: str | Path,
    output_dir: str | Path,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KMeans sobre patrón de flags posoperatorios + perfil preoperatorio por cluster.

    Parámetros:
        posop_path:   posop_raw.parquet
        merged_path:  merged.parquet de target_d_v2_hosp (para perfil preop)
        output_dir:   directorio de salida
        n_clusters:   número de clusters KMeans
        random_state: semilla

    Retorna:
        df_labels:  cluster + pca_1 + pca_2 por fila posoperatoria
        df_profile: prevalencia de flags + perfil preop por cluster
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_posop = pd.read_parquet(posop_path)
    flags_available = [f for f in _FLAGS_CLUSTERING if f in df_posop.columns]
    if not flags_available:
        raise ValueError("No clustering flags found in posop_raw.parquet")
    logger.info(f"Flags para clustering: {len(flags_available)}")

    X_flags = df_posop[flags_available].fillna(0).astype(float)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_flags)
    df_posop = df_posop.copy()
    df_posop["cluster"] = labels

    # ── Perfil de flags por cluster ───────────────────────────────────────────
    profile_rows = []
    for c in range(n_clusters):
        mask = df_posop["cluster"] == c
        row: dict = {"cluster": c, "n": int(mask.sum())}
        for flag in flags_available:
            row[flag] = round(float(df_posop.loc[mask, flag].mean()), 4)
        if "target" in df_posop.columns:
            row["target_prevalence"] = round(float(df_posop.loc[mask, "target"].mean()), 4)
        profile_rows.append(row)
    df_profile = pd.DataFrame(profile_rows).sort_values("cluster").reset_index(drop=True)

    # ── Perfil preoperatorio por cluster (desde merged, join por Documento PMD) ──
    df_merged = pd.read_parquet(merged_path)
    if "Edad" in df_merged.columns:
        df_merged = df_merged[df_merged["Edad"] >= 18]

    preop_cols = [c for c in _PREOP_PROFILE_COLS if c in df_merged.columns]

    posop_key = "Documento PMD (valoración preanestésica)"
    merged_key = "Documento PMD"

    if posop_key in df_posop.columns and merged_key in df_merged.columns:
        # Join por clave de paciente para evitar alineación posicional incorrecta
        df_posop_keys = df_posop[[posop_key, "cluster"]].rename(columns={posop_key: "_key"})
        df_merged_keys = df_merged[[merged_key] + preop_cols].rename(columns={merged_key: "_key"})
        df_joined = df_posop_keys.merge(df_merged_keys, on="_key", how="inner")

        if len(df_joined) > 100:
            preop_rows = []
            for c in range(n_clusters):
                sub = df_joined[df_joined["cluster"] == c]
                row = {"cluster": c}
                for col in preop_cols:
                    row[f"preop_{col}_mean"] = round(float(sub[col].mean()), 3)
                preop_rows.append(row)
            df_profile = df_profile.merge(pd.DataFrame(preop_rows), on="cluster", how="left")
            logger.info(f"Perfil preoperatorio por cluster incorporado ({len(df_joined)} pacientes).")
        else:
            logger.warning("Pocos matches en join por Documento PMD — perfil preop no incorporado.")
    else:
        logger.warning("Columna clave no disponible — perfil preop no incorporado.")

    # ── PCA 2D ────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=random_state)
    X_scaled = StandardScaler().fit_transform(X_flags)
    X_pca = pca.fit_transform(X_scaled)
    df_posop["pca_1"] = X_pca[:, 0]
    df_posop["pca_2"] = X_pca[:, 1]
    logger.info(f"PCA varianza explicada: {pca.explained_variance_ratio_.round(3)}")

    # ── Exportar CSVs ─────────────────────────────────────────────────────────
    df_labels = df_posop[
        ["cluster", "pca_1", "pca_2"]
        + (["target"] if "target" in df_posop.columns else [])
    ].copy()
    df_labels.to_csv(output_dir / "clustering_labels.csv", index=False)
    df_profile.to_csv(output_dir / "clustering_profile.csv", index=False)
    logger.info(f"Exportado: {output_dir / 'clustering_labels.csv'}")
    logger.info(f"Exportado: {output_dir / 'clustering_profile.csv'}")

    # ── PNG: heatmap + PCA scatter ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Heatmap
    heatmap_data = df_profile.set_index("cluster")[flags_available]
    heatmap_data.index = [f"C{i}" for i in heatmap_data.index]
    heatmap_data.columns = [c.replace("flag_", "") for c in heatmap_data.columns]
    sns.heatmap(
        heatmap_data.T, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Prevalencia de flags por cluster", fontweight="bold")
    axes[0].set_xlabel("Cluster")

    # PCA scatter (muestra para velocidad)
    sample = df_labels.sample(min(3000, len(df_labels)), random_state=random_state)
    palette = matplotlib.colormaps["tab10"]
    for c in range(n_clusters):
        mask = sample["cluster"] == c
        n_c = int((df_labels["cluster"] == c).sum())
        axes[1].scatter(
            sample.loc[mask, "pca_1"], sample.loc[mask, "pca_2"],
            c=[palette(c)], label=f"C{c} (n={n_c})",
            alpha=0.4, s=8,
        )
    axes[1].set_title("Clusters en espacio PCA (flags posoperatorios)", fontweight="bold")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8, markerscale=3)

    fig.suptitle("Estructura interna del dataset posoperatorio", fontweight="bold")
    plt.tight_layout()
    png_path = output_dir / "posop_clustering.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    logger.info(f"Exportado: {png_path}")

    for c in range(n_clusters):
        n = int((labels == c).sum())
        t = df_profile.loc[df_profile["cluster"] == c, "target_prevalence"].values
        t_str = f"  target={t[0]:.3f}" if len(t) else ""
        logger.info(f"  Cluster {c}: n={n}{t_str}")

    return df_labels, df_profile
