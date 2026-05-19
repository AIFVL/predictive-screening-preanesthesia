# Pipeline de Modelado — Documentación Completa

> **Proyecto:** Screening predictivo de valoración preanestésica Fundación Valle del Lili

## Objetivo del proyecto

El objetivo central es desarrollar un modelo de clasificación capaz de predecir, al momento de la valoración preanestésica, qué pacientes requieren una evaluación formal adicional antes de proceder a cirugía. El modelo opera exclusivamente sobre variables registradas durante la consulta preanestésica para anticipar la ocurrencia de complicaciones o eventos adversos posoperatorios.

Se trata de un problema de **clasificación binaria desbalanceada** en un entorno médico donde el coste de un falso negativo — no identificar a un paciente que sí requiere valoración — es sustancialmente mayor que el de un falso positivo — remitir a evaluación a un paciente que no la necesita. Esta asimetría condiciona todas las decisiones de umbral, métrica y calibración adoptadas en el proyecto.

---

## Estructura del pipeline

```
Datos brutos (Excel)
        │
        ▼
[1. Limpieza y preprocesamiento]   → output/v2/data_processed/preop_raw.parquet
        │                          → output/v2/data_processed/posop_raw.parquet
        │                          → output/v2/data_processed/cleaned.parquet
        │                          → output/v2/reports/cleaning_report.json
        │
        ▼
[2. Construcción del target]       → output/v2/data_processed/{target}/target_extracted.parquet
        │                          → Versiones activas: target_d_v2_hosp, target_f_predictibilidad_maxima
        │
        ▼
[3. Join preop + target]           → output/v2/data_processed/{target}/merged.parquet
        │
        ▼
[4. Selección de features]         → output/v2/data_processed/{target}/selected_features.json
        │                          → Top features por score combinado MI + RF
        │
        ▼
[5. División train/test]           → output/v2/data_processed/{target}/splits/
        │                          → 80/20, estratificado por target, random_state=42
        │
        ▼
[6. Entrenamiento de modelos]      → output/v2/models/{target}/{algoritmo}_model.joblib
        │                          → output/v2/models/{target}/{algoritmo}_metrics.json
        │                          → output/v2/models/{target}/{algoritmo}_manifest.json
        │
        ▼
[7a. evaluate]                     → output/v2/models/{target}/{algoritmo}_metrics.json
        │                          → output/v2/models/{target}/{algoritmo}_eval.json
        │                          → output/v2/plots/{target}/{algoritmo}_roc_pr.png
        │
        ▼
[7b. explainability]               → output/v2/reports/explainability/{target}/
        │
        ▼
[7c. calibration]                  → output/v2/reports/calibration/{target}/
        │
        ▼
[7d. shap_plots]                   → output/v2/reports/shap/{target}/shap_values_{algoritmo}.npy
        │                          → output/v2/reports/shap/{target}/shap_beeswarm_{algoritmo}.png
        │                          → output/v2/reports/shap/{target}/fn_waterfall/
        │
        ▼
[7e. shap_group_analysis]          → output/v2/reports/shap/{target}/group_analysis/
        │
        ▼
[8. API de inferencia]             → FastAPI sirve los manifests + modelos joblib
                                   → uvicorn api.main:app
```

---

## Documentos disponibles

### Pipeline de modelado

| Etapa                         | Documento                                                     |
| ----------------------------- | ------------------------------------------------------------- |
| Limpieza y preprocesamiento   | [01-limpieza-datos.md](01-limpieza-datos.md)                     |
| Construcción del target      | [02-construccion-target.md](02-construccion-target.md)           |
| Selección de features        | [03-seleccion-features.md](03-seleccion-features.md)             |
| Entrenamiento y evaluación   | [04-entrenamiento-evaluacion.md](04-entrenamiento-evaluacion.md) |
| Explicabilidad (importancias) | [05-explicabilidad.md](05-explicabilidad.md)                     |

### Aseguramiento de calidad clínico

| Tema                                              | Documento                           |
| ------------------------------------------------- | ----------------------------------- |
| Calibración de probabilidades                    | [06-calibracion.md](06-calibracion.md)                           |
| Análisis SHAP por grupos clínicos (TP/FN/FP/TN) | [07-shap-grupos.md](07-shap-grupos.md)                           |

### Servicio de inferencia

| Tema                            | Documento                                 |
| ------------------------------- | ----------------------------------------- |
| API REST (FastAPI) y despliegue | [08-api-inferencia.md](08-api-inferencia.md) |

### Análisis transversales

| Tema                                 | Documento                                                               |
| ------------------------------------ | ----------------------------------------------------------------------- |
| Análisis exploratorio posoperatorio | [../analisis-posoperatorio/README.md](../analisis-posoperatorio/README.md) |

---

## Fuentes de datos

El proyecto utiliza dos conjuntos de datos provenientes del sistema de información hospitalario de la Fundación Valle del Lili.

### Dataset preoperatorio (valoración preanestésica)

- **Origen:** Registros de consultas preanestésicas del sistema OPERA
- **Archivo bruto:** [`OPERA_PRE.xlsx`](../../OPERA_PRE.xlsx)
- **Procesado:** [`output/v2/data_processed/preop_raw.parquet`](../../output/v2/data_processed/preop_raw.parquet)
- **Filas brutas:** 30,962 registros
- **Filas tras filtrado de adultos (Edad ≥ 18):** 24,279 registros
- **Columnas brutas:** 236 variables
- **Clave de unión:** `Documento PMD` (identificador único de episodio)

### Dataset posoperatorio (registro de quirófano)

- **Origen:** Registros posoperatorios del sistema OPERA (hoja de anestesia)
- **Archivo bruto:** [`OPERA_POS.xlsx`](../../OPERA_POS.xlsx) y [`OPERA POSTQX.xlsx`](../../OPERA%20POSTQX.xlsx)
- **Procesado:** [`output/v2/data_processed/posop_raw.parquet`](../../output/v2/data_processed/posop_raw.parquet)
- **Filas:** 29,865 registros
- **Columnas:** 134 variables originales + flags clínicos derivados
- **Clave de unión:** `Documento PMD (valoración preanestésica)`

### Join entre datasets

- **Tipo:** INNER JOIN por `Documento PMD`
- **Resultado:** 23,387 pacientes con datos completos de ambas fuentes
- Reporte exacto en [`output/v2/reports/cleaning_report.json`](../../output/v2/reports/cleaning_report.json)

---

## Targets en producción (v2)

Tras múltiples iteraciones, el pipeline v2 entrena dos versiones del target:

| Slug API                 | `target_version` interno          | Lógica                                                          | Prevalencia | AUC mejor modelo  | Recomendado   |
| ------------------------ | ----------------------------------- | ---------------------------------------------------------------- | ----------- | ----------------- | ------------- |
| `general_risk`         | `target_d_v2_hosp`                | OR de flags clínicos refinados + hospitalización no anticipada | 27.69%      | ~0.761 (LightGBM) | No            |
| `hospitalization_risk` | `target_f_predictibilidad_maxima` | OR de los 5 flags más predecibles desde preop                   | 19.43%      | ~0.861 (XGBoost)  | **Sí** |

`target_f_predictibilidad_maxima` es la versión recomendada por tres razones: mayor predictibilidad desde variables preoperatorias (MI máxima 0.130 vs. 0.100, Pearson máximo 0.303 vs. 0.232), mejor AUC (~0.86 vs. ~0.76), y composición restringida a los flags con mayor señal individual: `flag_interconsultas`, `flag_hospitalizacion_no_anticipada`, `flag_uci_no_planeada`, `flag_estancia_uci` y `flag_estancia_prolongada`.

Los detalles de ambos targets y de las versiones históricas (target_a/b/c/d/e) se documentan en [02-construccion-target.md](02-construccion-target.md). La definición completa de cada versión reside en [`config/target_config.yaml`](../../config/target_config.yaml).

---

## Flujo de datos numérico (target seleccionado: `target_f_predictibilidad_maxima`)

```
30,962  registros preop brutos
 - 6,683  excluidos por edad < 18 años
= 24,279  en cleaned.parquet

24,279  registros limpios
× INNER JOIN con 29,865 posop
= 23,387  en merged.parquet (por Documento PMD)

23,387  con target_f_predictibilidad_maxima
├── ~4,544  positivos (19.43%) — necesitan valoración formal
└── ~18,843 negativos (80.57%)

División 80/20 estratificada (random_state=42):
├── Train: 18,709 registros
└── Test:   4,678 registros
```

---

## Modelos entrenados (v2)

Para cada target activo se entrenan **9 modelos** definidos en [`config/models_config.yaml`](../../config/models_config.yaml):

1. `logistic_regression` — Regresión logística (saga, max_iter=3000)
2. `random_forest` — Random Forest (300 árboles)
3. `extra_trees` — Extremely Randomized Trees (300 árboles)
4. `xgboost` — XGBoost (300 estimadores, lr=0.05, max_depth=6)
5. `hist_gradient_boosting` — HistGradientBoosting de sklearn (300 iter)
6. `lightgbm` — LightGBM (300 estimadores, num_leaves=31)
7. `mlp` — Red neuronal (capas 128 → 64, ReLU, early stopping)
8. `stacking` — Stacking (RF + XGB + HGB → meta-LR, cv=5)
9. `voting` — Voting soft (RF + XGB + HGB)

Todos los modelos compatibles se entrenan con `class_weight="balanced"` y se calibran con `CalibratedClassifierCV` cuando `calibrate: true` está configurado. Los umbrales de decisión se optimizan post-entrenamiento mediante la estrategia `recall_constraint`: se maximiza Precisión sujeto a la restricción Recall ≥ 0.85, según lo configurado en `pipeline_config.yaml`. Los detalles del entrenamiento y la evaluación se desarrollan en [04-entrenamiento-evaluacion.md](04-entrenamiento-evaluacion.md).

---

## Decisiones de diseño transversales

### 1. Métrica principal: F2 y Recall

En un contexto de screening médico, la consecuencia de un **falso negativo** — no identificar a un paciente que requiere valoración — es potencialmente grave. Por ello se adopta el **F2-score** como métrica de optimización del umbral, ponderando el Recall el doble que la Precisión. El Recall se monitoriza directamente como métrica secundaria, mientras que la AUC ROC y la AUC PR sirven como medidas globales de discriminación.

### 2. Umbrales bajos (0.13–0.18)

Los modelos de árbol calibrados operan con umbrales en el rango 0.13–0.18, clasificando como positivo a cualquier paciente con probabilidad ≥ 13–18%. Este tradeoff es deliberado: se aceptan más falsos positivos a cambio de mantener un Recall ≥ 0.85.

### 3. `class_weight="balanced"` y `scale_pos_weight`

Los modelos lineales y de árbol compatibles se entrenan con `class_weight="balanced"`. XGBoost emplea el parámetro equivalente `scale_pos_weight = n_neg / n_pos`.

### 4. Calibración isotónica

Los modelos basados en árboles producen probabilidades sesgadas, concentradas en los extremos o en rangos que no corresponden a frecuencias reales. Se aplica `CalibratedClassifierCV(method="isotonic")` para que la probabilidad de salida sea interpretable como riesgo real. La calidad de la calibración se evalúa con las métricas ECE, MCE y Brier — ver [06-calibracion.md](06-calibracion.md).

### 5. Imputación con valor centinela

Los valores faltantes de las features numéricas seleccionadas se imputan con `-1`, dado que todos los valores válidos del dataset son ≥ 0. Esta estrategia permite que el modelo aprenda la ausencia del dato como señal diferenciada. La estrategia exacta queda registrada en cada `manifest.json` bajo el campo `imputation`.

### 6. Múltiples versiones del target

Se evaluaron 9 versiones del target (a, b, c, d, d_v2, d_v2_hosp, d_v5, e, f). En v2 se conservaron únicamente las dos con mejor balance entre prevalencia, interpretabilidad clínica y rendimiento del modelo: `target_d_v2_hosp` y `target_f_predictibilidad_maxima`.

### 7. Análisis SHAP por grupo clínico

Más allá de las importancias globales, cada modelo se analiza con SHAP segmentando los casos por tipo de resultado clínico (TP/FN/FP/TN). Este análisis permite identificar los puntos ciegos del modelo — qué señales detecta en los falsos negativos que, sin embargo, no superan el umbral de decisión — y traducirlos en acciones concretas de mejora. Los detalles se desarrollan en [07-shap-grupos.md](07-shap-grupos.md).

### 8. Servicio de inferencia desacoplado

Junto al archivo `.joblib`, el pipeline de entrenamiento produce un `<algoritmo>_manifest.json` con el contrato completo del modelo: features, dtypes, threshold, estado de calibración e imputación. La API FastAPI consume esos manifests directamente, sin depender del código del pipeline. Cualquier modelo nuevo entrenado queda disponible para inferencia sin necesidad de modificar la API. Los detalles se documentan en [08-api-inferencia.md](08-api-inferencia.md).

---

## Reproducción del pipeline

### Setup inicial

```bash
# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Configurar variables de entorno para Airflow
export AIRFLOW__CORE__FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
export AIRFLOW_ADMIN_USER=admin
export AIRFLOW_ADMIN_PASSWORD=admin
export AIRFLOW_ADMIN_EMAIL=admin@example.com
```

### Ejecución con Docker Compose (recomendado)

Docker Compose inicia automáticamente:

- **Webserver de Airflow** (UI interactivo)
- **Scheduler de Airflow** (ejecutor de DAGs)
- **PostgreSQL** (backend de Airflow)
- **API FastAPI** (servicio de inferencia)

```bash
# Iniciar los servicios
docker compose up -d

# Verificar que todos los servicios estén sanos
docker compose ps

# Ver logs del scheduler
docker compose logs airflow-scheduler -f

# Ver logs de la API
docker compose logs api -f
```

### Acceso a la UI de Airflow

Una vez que los servicios estén disponibles, abrir **http://localhost:8080** en el navegador e iniciar sesión con las credenciales configuradas (por defecto: `admin` / `admin`). Desde la lista de DAGs, localizar **`preanesthesia_pipeline`** y activarlo con el botón **"Trigger DAG"** para ejecutar el pipeline completo.

El DAG ejecuta automáticamente todas las etapas en orden: validación de datos crudos, EDA preoperatorio, limpieza y enriquecimiento de datos, extracción del target para cada versión activa, join preop + target, selección de features, EDA posoperatorio y correlaciones, entrenamiento de 9 modelos, evaluación, calibración, análisis SHAP, explicabilidad y generación de reportes comparativos.

### Monitoreo del pipeline

La UI de Airflow ofrece distintas vistas para monitorear la ejecución: **Graph View** para visualizar las dependencias entre tareas, **Tree View** para consultar el historial de ejecuciones, **Logs** para inspeccionar la salida detallada de cada tarea, y **Gantt Chart** para analizar la duración de cada etapa.

### Acceso a la API de inferencia

Una vez que el pipeline termina, la API está disponible:

```bash
# Health check
curl http://localhost:8000/health

# Documentación interactiva (Swagger UI)
http://localhost:8000/docs

# Ejemplo de predicción
curl -X POST http://localhost:8000/models/{target}/{algorithm}/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

Los modelos entrenados se cargan automáticamente desde `output/v2/models/` en la API.

### Detener los servicios

```bash
# Detener todo (sin eliminar volúmenes)
docker compose down

# Detener y eliminar volúmenes (ATENCION: borra logs de Airflow y DB)
docker compose down -v
```
