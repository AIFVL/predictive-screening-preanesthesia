# Análisis Exploratorio Posoperatorio

> **Fecha de ejecución:** 2026-04-05
> **Rama git:** `feat/refactor-pipeline`
> **Commits relevantes:** `e1a9df7` (package), `cbf5aa3` (flag_predictability), `b22dcab` (error_subgroups), `0801302` (posop_clustering), `00cd755` (DAG), `3af2beb` (fix posop join)

---

## Contexto y motivación

El proyecto busca predecir qué pacientes de la Fundación Valle del Lili requieren valoración preanestésica formal antes de una cirugía. Se entrenaron modelos de clasificación sobre variables preoperatorias (valoración preanestésica) para predecir un **target compuesto** que agrega múltiples tipos de complicaciones o eventos posoperatorios.

El problema identificado: el modelo alcanza métricas moderadas (AUC ~0.75, Recall ~0.86) pero se sospecha que el **target compuesto** — al mezclar tipos de complicaciones muy distintos — limita artificialmente la capacidad predictiva. Algunos eventos posoperatorios pueden ser predecibles desde variables preoperatorias; otros ocurren por razones intraoperatorias incontrolables.

Este análisis exploratorio se ejecutó para responder tres preguntas independientes:

| Enfoque | Pregunta | Documento |
|---------|----------|-----------|
| **C** | ¿Qué flags posoperatorios individuales son predecibles desde datos preoperatorios? | [enfoque-C-flag-predictability.md](enfoque-C-flag-predictability.md) |
| **B** | ¿En qué subgrupos de pacientes el modelo funciona mejor o peor? | [enfoque-B-error-subgroups.md](enfoque-B-error-subgroups.md) |
| **A** | ¿Qué estructuras internas (tipos de paciente) existen en el dataset posoperatorio? | [enfoque-A-posop-clustering.md](enfoque-A-posop-clustering.md) |

---

## Datos utilizados

| Dataset | Ruta | Descripción |
|---------|------|-------------|
| `merged.parquet` | `output/v1/data_processed/target_d_v2_hosp/merged.parquet` | 23,387 registros con variables preoperatorias y target. Filtrado a mayores de 18 años. |
| `posop_raw.parquet` | `output/v1/data_processed/posop_raw.parquet` | 29,865 registros posoperatorios con 57 flags clínicos binarios. |
| `X_test.parquet` | `output/v1/data_processed/target_d_v2_hosp/splits/X_test.parquet` | 4,678 registros del conjunto de prueba. |
| `y_test.parquet` | `output/v1/data_processed/target_d_v2_hosp/splits/y_test.parquet` | Labels del conjunto de prueba (27.7% positivos). |
| `random_forest_model.joblib` | `output/v1/models/target_d_v2_hosp/random_forest_model.joblib` | Modelo Random Forest entrenado sobre `target_d_v2_hosp`. |
| `random_forest_metrics.json` | `output/v1/models/target_d_v2_hosp/random_forest_metrics.json` | Métricas del modelo en test. |

### Métricas del modelo de referencia (Random Forest, target_d_v2_hosp)

| Métrica | Valor |
|---------|-------|
| ROC AUC | 0.7593 |
| Recall | 0.8600 |
| Precision | 0.3537 |
| F1 | 0.5013 |
| F2 | 0.6686 |
| Specificity | 0.3984 |
| Threshold | 0.17 |

> El threshold de 0.17 fue elegido para maximizar Recall (detectar la mayor cantidad posible de pacientes que sí necesitan valoración). A ese umbral, el modelo clasifica el 67.3% de los pacientes como positivos — un tradeoff consciente entre sensibilidad y especificidad.

---

## Outputs generados

Todos los outputs están en `output/v1/reports/analisis_posoperatorio/`:

| Archivo | Generado por | Contenido |
|---------|-------------|-----------|
| `flag_predictability.csv` | `src/analysis/flag_predictability.py` | ROC AUC por flag, prevalencia, interpretación |
| `flag_predictability.png` | `src/analysis/flag_predictability.py` | Gráfico de barras horizontales por flag |
| `error_analysis.csv` | `src/analysis/error_subgroups.py` | AUC, FN rate, FP rate por subgrupo |
| `error_subgroups.png` | `src/analysis/error_subgroups.py` | Grilla 2×2 de AUC por variable de segmentación |
| `clustering_labels.csv` | `src/analysis/posop_clustering.py` | Cluster asignado + coordenadas PCA por paciente |
| `clustering_profile.csv` | `src/analysis/posop_clustering.py` | Prevalencia de flags y perfil preop por cluster |
| `posop_clustering.png` | `src/analysis/posop_clustering.py` | Heatmap + scatter PCA |

---

## Cómo ejecutar

Los tres análisis se pueden correr desde el DAG de Airflow (`dags/analysis_pipeline.py`, `dag_id="analysis_pipeline"`, `schedule=None`) o directamente en Python:

```python
# Enfoque C — Predictibilidad de flags
from src.analysis.flag_predictability import run_flag_predictability
run_flag_predictability(
    merged_path="output/v1/data_processed/target_d_v2_hosp/merged.parquet",
    posop_path="output/v1/data_processed/posop_raw.parquet",
    output_dir="output/v1/reports/analisis_posoperatorio",
)

# Enfoque B — Error por subgrupos
from src.analysis.error_subgroups import run_error_subgroups
run_error_subgroups(
    merged_path="output/v1/data_processed/target_d_v2_hosp/merged.parquet",
    splits_dir="output/v1/data_processed/target_d_v2_hosp/splits",
    model_path="output/v1/models/target_d_v2_hosp/random_forest_model.joblib",
    output_dir="output/v1/reports/analisis_posoperatorio",
    threshold=0.17,
)

# Enfoque A — Clustering posoperatorio
from src.analysis.posop_clustering import run_posop_clustering
run_posop_clustering(
    posop_path="output/v1/data_processed/posop_raw.parquet",
    merged_path="output/v1/data_processed/target_d_v2_hosp/merged.parquet",
    output_dir="output/v1/reports/analisis_posoperatorio",
    n_clusters=5,
)
```

---

## Conclusiones integradas

Los tres enfoques convergen en el mismo diagnóstico:

1. **El target compuesto mezcla complicaciones predecibles con impredecibles.** El Enfoque C demuestra que algunos flags tienen AUC individual > 0.80 mientras que otros no superan 0.60. Al combinarlos en un solo target, el modelo aprende una señal promediada hacia abajo.

2. **Existen subgrupos de pacientes donde el modelo falla sistemáticamente.** El Enfoque B muestra AUC de 0.54 en pacientes endocrinológicos — los que más se beneficiarían de valoración formal. El modelo es más preciso donde es menos necesario.

3. **Las complicaciones posoperatorias no son un fenómeno homogéneo.** El Enfoque A identifica 5 clusters con perfiles clínicamente distintos. Los clusters 1–4 (100% target positivo) representan fenotipos de complicación completamente diferentes entre sí.

**Implicación directa:** Redefinir el target hacia complicaciones de alto impacto y alta predictibilidad (`hospitalizacion_no_anticipada`, `uci_no_planeada`, `estancia_uci`, `interconsultas`) podría producir un modelo significativamente más preciso y clínicamente más interpretable.
