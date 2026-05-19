# Modelo Predictivo Pre-Anestésico

Sistema de aprendizaje automático de extremo a extremo para la predicción de eventos adversos perioperatorios en pacientes quirúrgicos, a partir de registros clínicos preoperatorios.

---

## Tabla de Contenidos

1. [Propósito del Proyecto](#1-propósito-del-proyecto)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Requisitos Previos](#3-requisitos-previos)
4. [Instalación y Configuración](#4-instalación-y-configuración)
5. [Ejecución del Pipeline Completo](#5-ejecución-del-pipeline-completo)
6. [Uso de la API de Inferencia](#6-uso-de-la-api-de-inferencia)
7. [Referencia de Configuración](#7-referencia-de-configuración)
8. [Pruebas Automatizadas](#8-pruebas-automatizadas)
9. [Consideraciones Importantes](#9-consideraciones-importantes)

---

## 1. Propósito del Proyecto

Este proyecto implementa un pipeline de aprendizaje automático de producción para el tamizaje predictivo preanestésico. A partir de registros clínicos preoperatorios —datos demográficos, diagnósticos, medicamentos, valores de laboratorio y metadata del procedimiento— el sistema entrena y sirve modelos que estiman la probabilidad de eventos adversos perioperatorios, permitiendo a los anestesiólogos estratificar el riesgo y priorizar la evaluación antes de la cirugía.

**¿Qué hace el sistema en concreto?**

- **Entrena modelos automáticamente:** Apache Airflow orquesta el pipeline completo —validación, limpieza (incluida normalización de medicamentos y diagnósticos vía NLP), ingeniería de características, selección de variables, entrenamiento de hasta 9 familias de modelos con calibración probabilística y optimización de umbral— y guarda los resultados en `output/v2/`.
- **Sirve predicciones vía API REST:** Un servidor FastAPI carga los modelos entrenados y expone un esquema dinámico por modelo. Se envía JSON con datos del paciente y se obtiene un score de riesgo calibrado, junto con nivel de riesgo (`low`, `medium`, `high`).
- **Corre completamente en Docker:** No se requiere instalación local de Python. Un solo comando levanta toda la infraestructura.

**Dos variables objetivo están disponibles:**

| Nombre en la API         | Target interno                      | Descripción                                                                                              | Prevalencia |
| ------------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------- |
| `hospitalization_risk` | `target_f_predictibilidad_maxima` | Hospitalización no anticipada, UCI no planeada, estancia prolongada (lógica OR —**recomendado**) | ~27.7%      |
| `general_risk`         | `target_d_v2_hosp`                | Eventos adversos perioperatorios compuestos + hospitalización inesperada                                 | ~25.8%      |

> [!NOTE]
> Este es un proyecto académico de investigación. Las predicciones del modelo son señales de tamizaje y **no reemplazan el juicio clínico**. El sistema está calibrado para minimizar falsos negativos (`recall_min = 0.85`), lo que implica una tasa de falsos positivos elevada.

---

## 2. Estructura del Repositorio

```
predictive-screening-preanesthesia/
│
├── api/                           # Servidor de inferencia FastAPI
│   ├── main.py                    # Punto de entrada, lifespan, registro de rutas
│   ├── core/                      # Configuración, logging, parches de compatibilidad sklearn
│   ├── domain/                    # Registro de modelos y tipos de dominio (manifests)
│   ├── routers/                   # Rutas: /health, /targets, /models, /predict
│   ├── schemas/                   # Generación dinámica de esquemas Pydantic por modelo
│   └── services/                  # Lógica de predicción, preprocesamiento, estratificación
│
├── src/                           # Librería Python del pipeline (paquete importable)
│   ├── cleaning/                  # Limpieza de datos, detección de outliers, enriquecimiento NLP
│   ├── data/                      # Cargadores y validadores de datos crudos
│   ├── datasets/                  # División train/test/CV, mezcla de datasets
│   ├── target/                    # Construcción de variables objetivo desde flags posoperatorios
│   ├── features/                  # Ingeniería de características, codificación, poda por varianza
│   ├── models/                    # Entrenamiento, calibración, manifests, búsqueda Optuna
│   ├── evaluation/                # Métricas (F2, ROC-AUC), calibración, SHAP, subgrupos
│   ├── reports/                   # Generación de reportes EDA, curvas, explainability CSV
│   ├── analysis/                  # Análisis post-pipeline (clustering posop, errores por subgrupo)
│   └── utils/                     # I/O, logger, config loader, versionado
│
├── dags/
│   ├── preanesthesia_pipeline.py  # DAG principal de Airflow (pipeline completo)
│   └── analysis_pipeline.py      # DAG de análisis posoperatorio independiente
│
├── config/
│   ├── pipeline_config.yaml       # Versión, rutas, split, CV, optimización de umbral
│   ├── models_config.yaml         # Definición de modelos e hiperparámetros
│   ├── target_config.yaml         # 9 definiciones de targets, 2 activos
│   ├── features_config.yaml       # Lista de features, mapa de encoding, poda y selección
│   └── cleaning_config.yaml       # Columnas identificadoras, filtro de edad, reglas de outliers
│
├── notebooks/                     # Notebooks secuenciales de EDA y modelado (01–07)
│
├── scripts/
│   └── backfill_model_manifests.py  # Utilidad para regenerar manifests de modelos guardados
│
├── tests/
│   ├── src/                       # Tests unitarios del pipeline (cleaning, models, features...)
│   ├── utils/                     # Tests de utilidades compartidas
│   └── api/                       # Tests de integración y E2E para FastAPI
│
├── docs/                          # Documentación técnica del pipeline (pasos 01–08, en español)
│
├── output/
│   └── v2/
│       ├── data_processed/        # Datasets limpios y con features engineered (.parquet)
│       ├── models/                # Modelos serializados (.joblib) y manifests (.json)
│       ├── plots/                 # Gráficas de EDA y evaluación
│       └── reports/
│           ├── calibration/       # Diagnósticos de calibración por modelo y target
│           ├── shap/              # Valores SHAP globales
│           └── explainability/    # Contribuciones SHAP por caso (CSV)
│
├── data/                          # Datos crudos — NO incluidos en el repositorio (.gitkeep)
├── docker-compose.yaml            # Orquestación completa de servicios
├── Dockerfile                     # Imagen principal para servicios Airflow
├── .env.example                   # Plantilla de variables de entorno
└── requirements.txt               # Dependencias Python del pipeline
```

---

## 3. Requisitos Previos

### Software requerido

| Requisito      | Versión mínima | Notas                                           |
| -------------- | ---------------- | ----------------------------------------------- |
| Docker Desktop | 4.x              | Con backend**WSL2** habilitado en Windows |
| Docker Compose | 2.20+            | Incluido en Docker Desktop                      |
| Python         | 3.11             | Solo para desarrollo local fuera de Docker      |

> [!WARNING]
> **Python 3.11 es estrictamente requerido.** La imagen base de Airflow (`apache/airflow:2.9.1-python3.11`) y el Dockerfile de la API (`python:3.11-slim`) apuntan exclusivamente a esta versión. No se ha validado con Python 3.10 ni 3.12+. Se recomienda siempre usar el entorno Docker para evitar problemas de compatibilidad.

### Recursos de hardware recomendados

| Recurso                | Mínimo                                         | Recomendado |
| ---------------------- | ----------------------------------------------- | ----------- |
| RAM asignada a Docker  | 8 GB                                            | 12 GB       |
| Espacio en disco libre | 10 GB                                           | 20 GB       |
| Acceso a internet      | Requerido (build inicial y enriquecimiento NLP) | —          |

> [!IMPORTANT]
> **Usuarios de Windows:** Docker Desktop debe tener el backend **WSL2** habilitado. El backend Hyper-V no ha sido validado y puede presentar problemas con los bind mounts de directorios.

**Distribución aproximada del espacio en disco:**

- Imagen `apache/airflow:2.9.1`: ~1.5 GB
- Caché de modelos HuggingFace (volumen `hf_cache`): hasta 5 GB
- Outputs del pipeline (`output/v2/`): 1–3 GB por ejecución completa

---

## 4. Instalación y Configuración

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/AIFVL/predictive-screening-preanesthesia.git
cd predictive-screening-preanesthesia
```

### Paso 2 — Crear y configurar el archivo de entorno

```bash
cp .env.example .env
```

Generar una clave Fernet para el cifrado de la base de datos de metadatos de Airflow:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Editar `.env` y completar los valores:

```dotenv
AIRFLOW__CORE__FERNET_KEY=<clave-generada-aquí>
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=<contraseña-segura>
AIRFLOW_ADMIN_EMAIL=admin@example.com
```

> [!IMPORTANT]
> **Guardar la clave Fernet de forma segura.** Esta clave cifra las credenciales almacenadas en la base de datos de Airflow. Si se pierde o se cambia después de que existan ejecuciones registradas, la base de metadatos quedará inaccesible. El archivo `.env` **no debe ser comprometido a control de versiones**.

### Paso 3 — Colocar los datos crudos

El directorio `data/` está vacío (solo contiene `.gitkeep`). Se deben colocar los archivos clínicos fuente antes de ejecutar el pipeline:

```
data/
├── preop_records.csv      # Registros clínicos preoperatorios
└── posop_records.csv      # Registros de resultados posoperatorios
```

> [!IMPORTANT]
> **Los datos no están incluidos en este repositorio.** Los archivos deben tener exactamente los nombres de columnas definidos en `config/features_config.yaml` y `config/cleaning_config.yaml`. La columna identificadora esperada es `CODIGO` y la columna de edad es `Edad`. Cualquier discrepancia en nombres de columna resultará en fallos silenciosos (columnas tratadas como NaN).

### Paso 4 — Construir e iniciar todos los servicios

```bash
docker-compose up --build
```

Este comando inicia cinco servicios en orden de dependencia:

| Servicio              | Puerto | Descripción                                                     |
| --------------------- | ------ | ---------------------------------------------------------------- |
| `postgres`          | 5432   | Backend de metadatos de Airflow (PostgreSQL 15)                  |
| `airflow-init`      | —     | Migraciones de BD + creación de usuario admin (ejecuta una vez) |
| `airflow-webserver` | 8080   | Interfaz web de Airflow                                          |
| `airflow-scheduler` | —     | Planificación y ejecución de tareas DAG                        |
| `api`               | 8000   | Servidor de inferencia FastAPI                                   |

Esperar a que `airflow-init` termine con código 0 antes de interactuar con la UI. El servidor API realiza una verificación de salud al iniciar; si no hay modelos entrenados en `output/v2/models/`, los endpoints de predicción devuelven 503 hasta que el pipeline haya corrido.

> [!NOTE]
> Si se modifican archivos bajo `config/` o `src/`, los contenedores en ejecución reciben los cambios de inmediato (están montados como volúmenes bind). Solo se necesita reconstruir (`--build`) cuando cambian `requirements.txt`, `Dockerfile` o `api/Dockerfile`.

---

## 5. Ejecución del Pipeline Completo

### Acceder a la interfaz de Airflow

Navegar a [http://localhost:8080](http://localhost:8080) e iniciar sesión con las credenciales configuradas en `.env`.

### Habilitar y ejecutar el DAG principal

1. Localizar el DAG **`preanesthesia_pipeline`** en la lista de DAGs.
2. Activarlo con el interruptor de **On/Off** (despausarlo).
3. Hacer clic en **Trigger DAG** para iniciar una ejecución manual.

El DAG ejecuta la siguiente secuencia de tareas (por cada target activo):

```
validate_raw_data
    └── eda_preop_raw
        └── clean_data               ← limpieza + enriquecimiento NLP
            └── eda_preop_clean
                └── extract_target_{target}
                    └── merge_datasets_{target}
                        └── eda_posop_{target}
                            └── split_datasets_{target}
                                └── select_features_{target}
                                    └── train_models_{target}    ← paralelo por modelo
                                        └── evaluate_models_{target}
                                            └── generate_reports_{target}
```

**Parámetros clave del pipeline** (`config/pipeline_config.yaml`):

```yaml
pipeline_version: "v2"
train_test_split:
  test_size: 0.2
  random_state: 42
cross_validation:
  n_folds: 10
threshold:
  optimize_for: "recall_constraint"
  recall_min: 0.85
hyperparameter_search:
  enabled: false      # Habilitar agrega horas al entrenamiento (70 trials Optuna por modelo)
  n_iter: 70
  scoring: "f2"
```

**Monitoreo de tareas:** Los logs de cada tarea son accesibles desde la UI de Airflow (vista Grid → seleccionar tarea → pestaña Logs). Los outputs del pipeline se escriben en `output/v2/` sobre el volumen Docker y están disponibles en el sistema de archivos del host.

### Modelos disponibles

El pipeline entrena hasta 9 familias de modelos (configurables en `config/models_config.yaml`):

| Modelo                     | Clase base                                          |
| -------------------------- | --------------------------------------------------- |
| `logistic_regression`    | `sklearn.linear_model.LogisticRegression`         |
| `random_forest`          | `sklearn.ensemble.RandomForestClassifier`         |
| `extra_trees`            | `sklearn.ensemble.ExtraTreesClassifier`           |
| `xgboost`                | `xgboost.XGBClassifier`                           |
| `hist_gradient_boosting` | `sklearn.ensemble.HistGradientBoostingClassifier` |
| `lightgbm`               | `lightgbm.LGBMClassifier`                         |
| `mlp`                    | `sklearn.neural_network.MLPClassifier`            |
| `stacking`               | RF + XGB + HGB → LR meta-learner                   |
| `voting`                 | Votación suave RF + XGB + HGB                      |

Todos los modelos se calibran con Platt scaling y se les optimiza el umbral de decisión con restricción `recall >= 0.85`.

---

## 6. Uso de la API de Inferencia

El servidor FastAPI está disponible en [http://localhost:8000](http://localhost:8000). La documentación interactiva (Swagger UI) se encuentra en [http://localhost:8000/docs](http://localhost:8000/docs).

El servidor escanea `output/v2/models/` al inicio y registra todos los targets y algoritmos para los cuales exista un `*_manifest.json` válido. Los esquemas de entrada Pydantic se generan dinámicamente por modelo desde el manifest.

### Referencia de endpoints

| Método  | Ruta                                    | Descripción                                              |
| -------- | --------------------------------------- | --------------------------------------------------------- |
| `GET`  | `/health`                             | Verificación de disponibilidad del servicio              |
| `GET`  | `/targets`                            | Lista de targets de predicción disponibles               |
| `GET`  | `/models?target={target}`             | Lista de algoritmos disponibles para un target            |
| `GET`  | `/models/{target}/{algorithm}/schema` | Esquema de features de entrada para un modelo específico |
| `POST` | `/predict/{target}/{algorithm}`       | Predicción para un único paciente                       |
| `POST` | `/predict/{target}/{algorithm}/batch` | Predicción en lote (hasta 100 registros)                 |

### Ejemplos con curl

**Verificar disponibilidad:**

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

**Listar targets disponibles:**

```bash
curl -s http://localhost:8000/targets | python -m json.tool
```

**Listar modelos para un target:**

```bash
curl -s "http://localhost:8000/models?target=hospitalization_risk" | python -m json.tool
```

**Consultar el esquema de features de un modelo:**

```bash
curl -s http://localhost:8000/models/hospitalization_risk/logistic_regression/schema \
  | python -m json.tool
```

**Predicción individual:**

```bash
curl -s -X POST http://localhost:8000/predict/hospitalization_risk/logistic_regression \
  -H "Content-Type: application/json" \
  -d '{
    "Edad": 62,
    "Peso (Kg)": 78,
    "Talla (cm)": 168,
    "IMC": 27.6,
    "Tensión Arterial Sistólica (mm/Hg)": 140,
    "Frecuencia Cardíaca (lpm)": 82,
    "Saturación de Oxígeno (%)": 97
  }' | python -m json.tool
```

> [!NOTE]
> **Todos los campos son opcionales.** Los valores faltantes se imputan automáticamente con la mediana del conjunto de entrenamiento. Se recomienda incluir al menos: edad, IMC, signos vitales y tipo de anestesia propuesta para obtener predicciones más confiables.

**Predicción en lote:**

```bash
curl -s -X POST http://localhost:8000/predict/hospitalization_risk/random_forest/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"Edad": 62, "IMC": 27.6, "Tensión Arterial Sistólica (mm/Hg)": 140},
    {"Edad": 45, "IMC": 30.1}
  ]' | python -m json.tool
```

### Estructura de la respuesta

```json
{
  "success": true,
  "data": {
    "predicted_class": 1,
    "probability": 0.73,
    "threshold": 0.19,
    "risk_level": "high",
    "calibrated": true,
    "prevalence_train": 0.258,
    "warnings": []
  },
  "meta": {
    "request_id": "a3f1c2...",
    "model_id": "target_f_predictibilidad_maxima__logistic_regression"
  }
}
```

| Campo               | Descripción                                                         |
| ------------------- | -------------------------------------------------------------------- |
| `predicted_class` | `1` = paciente en riesgo, `0` = menor riesgo                     |
| `probability`     | Probabilidad calibrada del evento adverso (0.0–1.0)                 |
| `threshold`       | Umbral de decisión aplicado (optimizado para F2,`recall >= 0.85`) |
| `risk_level`      | Nivel de riesgo estratificado:`"low"`, `"medium"` o `"high"`   |
| `calibrated`      | Si se aplicó calibración de probabilidades (Platt scaling)         |
| `warnings`        | Alertas de calidad para esta predicción                             |

---

## 7. Referencia de Configuración

Toda la lógica del pipeline está controlada por cinco archivos YAML en `config/`. No se requieren cambios en el código para ajustar el comportamiento estándar.

**`config/pipeline_config.yaml`** — Configuración maestra del pipeline:

```yaml
pipeline_version: "v2"
train_test_split: { test_size: 0.2, random_state: 42 }
cross_validation: { n_folds: 10, n_jobs: -1 }
threshold: { optimize_for: "recall_constraint", recall_min: 0.85 }
hyperparameter_search: { enabled: false, n_iter: 70, scoring: "f2", cv_folds: 10 }
```

**`config/models_config.yaml`** — Define las familias de modelos, hiperparámetros y si se aplica calibración. Todos los modelos tienen `calibrate: true` y `target_metric: "f2"`.

**`config/target_config.yaml`** — Define hasta 9 targets con lógica de subflag (AND/OR/composite). Los targets activos son `target_d_v2_hosp` y `target_f_predictibilidad_maxima`. Para cambiar los targets activos, modificar el campo `active: true/false` en este archivo.

**`config/features_config.yaml`** — Controla ingeniería y selección de features:

```yaml
numerical_features: [Edad, Peso, Talla, IMC, ...]
feature_pruning: { min_variance: 0.01 }
feature_selection: { min_combined_score: 0.02 }  # Retiene ~117 de 234 features
encoding_fix_map: { ... }  # Correcciones de mojibake en nombres de columna
```

**`config/cleaning_config.yaml`** — Reglas de limpieza:

```yaml
identifier_columns: [CODIGO]
age_filter: { column: "Edad", min: 18 }
outlier_rules: { Edad: { min: 0, max: 120 } }
```

---

## 8. Pruebas Automatizadas

La suite de pruebas usa **pytest** y cubre tests unitarios de módulos del pipeline, tests de utilidades y tests de integración/E2E completos para el servidor FastAPI.

**Instalar dependencias de prueba** (entorno local, Python 3.11):

```bash
pip install -r api/requirements.txt
pip install pytest pytest-cov httpx
```

**Ejecutar la suite completa:**

```bash
pytest tests/ -v
```

**Ejecutar por alcance:**

```bash
# Solo tests unitarios del pipeline
pytest tests/src/ -v

# Tests de utilidades compartidas
pytest tests/utils/ -v

# Tests de integración y E2E de la API
pytest tests/api/ -v
```

**Con reporte de cobertura:**

```bash
pytest tests/ --cov=src --cov=api --cov-report=term-missing
```

> [!NOTE]
> Los tests E2E en `tests/api/` que ejercen el path completo de predicción requieren que los artefactos de modelo existan en `output/v2/models/`. Si se ejecutan antes de que el pipeline haya corrido, los tests que dependen de modelos cargados serán omitidos o devolverán respuestas 503 — este es el comportamiento esperado. Los tests de esquema y ruteo no requieren modelos.

---

## 9. Consideraciones Importantes

### Descargo médico

> [!WARNING]
> Este es un proyecto académico de investigación. Las predicciones del sistema **no deben reemplazar el juicio clínico**. El modelo está optimizado para minimizar falsos negativos (`recall_min = 0.85`): un paciente marcado como "alto riesgo" no tiene un diagnóstico; la salida es una señal de tamizaje orientada a priorizar la revisión clínica, no una decisión médica.

### Restricciones de versiones de dependencias

> [!WARNING]
> **`numpy` debe permanecer por debajo de 2.0.0.** Ambos `requirements.txt` y `api/requirements.txt` fuerzan `numpy>=1.24.0,<2.0.0`. Los archivos `.joblib` de los modelos fueron serializados con scikit-learn dependiendo de los internals de pickle de numpy 1.x. Instalar `numpy>=2.0.0` causará errores de deserialización al cargar modelos. No actualizar numpy sin reentrenar todos los modelos desde cero.

> [!WARNING]
> **Desajuste de versiones de scikit-learn entre entrenamiento e inferencia.** El entorno de entrenamiento usa `scikit-learn>=1.3.0` mientras que la API requiere `>=1.7.0,<2.0.0`. El parche en `api/core/sklearn_compat.py` aborda incompatibilidades de `__sklearn_tags__` para XGBoost y LightGBM, pero no cubre todos los posibles problemas de serialización entre versiones. El campo `sklearn_version` en cada `*_manifest.json` registra la versión usada en entrenamiento.

### Seguridad: CVE conocido en torch

> [!WARNING]
> La dependencia `torch>=2.6.0` tiene una vulnerabilidad conocida (CVE-2025-32434). `torch` se usa exclusivamente para normalización de texto en el paso de limpieza; **no interviene en el entrenamiento ni en la inferencia principal**. `safetensors>=0.4.0` se incluye específicamente para cargar pesos sin usar `torch.load`, que es el path afectado. En entornos de producción regulados, evaluar si la dependencia puede ser aislada hasta que exista un parche.

### Sobrescritura de manifests al reentrenar

Cada modelo guarda su `manifest.json` junto al artefacto `.joblib` bajo `output/v2/models/<target>/`. Reentrenar **sobrescribe** estos archivos. No hay versionado de modelos integrado más allá del sistema de archivos. Si se necesita preservar métricas y metadatos de un entrenamiento anterior, respaldar `output/v2/models/` antes de reentrenar.

### Búsqueda de hiperparámetros deshabilitada por defecto

La optimización con Optuna está en `enabled: false` en `config/pipeline_config.yaml`. Habilitarla agrega hasta 70 trials por modelo por target (10-fold CV, métrica F2), lo que puede añadir **varias horas** a una ejecución completa en entornos con recursos limitados.

### Enriquecimiento NLP externo

El paso de limpieza invoca modelos de transformers para normalización de medicamentos y diagnósticos. Esto requiere acceso a internet para descargar pesos de HuggingFace en el primer uso y agrega latencia significativa a la tarea de limpieza del DAG. En entornos sin acceso a internet, el volumen `hf_cache` debe ser pre-poblado manualmente.

---

## Documentación Adicional

La carpeta [`docs/`](docs/) contiene documentación detallada de cada paso del pipeline:

| Documento                                                                                 | Contenido                              |
| ----------------------------------------------------------------------------------------- | -------------------------------------- |
| [docs/pipeline/01-limpieza-datos.md](docs/pipeline/01-limpieza-datos.md)                     | Limpieza de datos preoperatorios       |
| [docs/pipeline/02-construccion-target.md](docs/pipeline/02-construccion-target.md)           | Construcción de variables objetivo    |
| [docs/pipeline/03-seleccion-features.md](docs/pipeline/03-seleccion-features.md)             | Selección de características         |
| [docs/pipeline/04-entrenamiento-evaluacion.md](docs/pipeline/04-entrenamiento-evaluacion.md) | Entrenamiento y evaluación de modelos |
| [docs/pipeline/05-explicabilidad.md](docs/pipeline/05-explicabilidad.md)                     | Explicabilidad (SHAP, importancias)    |
| [docs/pipeline/06-calibracion.md](docs/pipeline/06-calibracion.md)                           | Calibración de probabilidades         |
| [docs/pipeline/07-shap-grupos.md](docs/pipeline/07-shap-grupos.md)                           | Análisis SHAP por subgrupos           |
| [docs/pipeline/08-api-inferencia.md](docs/pipeline/08-api-inferencia.md)                     | API de inferencia                      |

---

## Tecnologías Principales

| Capa            | Tecnologías                                                 |
| --------------- | ------------------------------------------------------------ |
| Orquestación   | Apache Airflow 2.9.1, PostgreSQL 15                          |
| Pipeline ML     | pandas ≥ 2.0, numpy ≥ 1.24,<2.0, scikit-learn ≥ 1.3       |
| Modelos         | XGBoost ≥ 2.0, LightGBM ≥ 4.0, scikit-learn ensembles, MLP |
| Optimización   | Optuna ≥ 3.0                                                |
| Explicabilidad  | SHAP ≥ 0.44                                                 |
| NLP (limpieza)  | torch ≥ 2.6, transformers ≥ 4.30, HuggingFace Hub          |
| API             | FastAPI ≥ 0.110, Uvicorn ≥ 0.29, Pydantic ≥ 2.5           |
| Testing         | pytest ≥ 7.4, httpx ≥ 0.27                                 |
| Infraestructura | Docker, Docker Compose, Python 3.11                          |
