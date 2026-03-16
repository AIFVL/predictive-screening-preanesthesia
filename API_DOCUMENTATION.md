# Preanesthesia Screening API — Documentación Técnica

**Versión:** 1.0.0
**Stack:** FastAPI · scikit-learn · XGBoost · SHAP · Docker
**Entrypoint:** `api/main.py`
**Docs interactivos:** `http://localhost:8000/docs`

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Arquitectura](#2-arquitectura)
3. [Formato del Artefacto de Modelo](#3-formato-del-artefacto-de-modelo)
4. [Instalación y Puesta en Marcha](#4-instalación-y-puesta-en-marcha)
5. [Referencia de Endpoints](#5-referencia-de-endpoints)
6. [Flujo Completo de Uso](#6-flujo-completo-de-uso)
7. [Entrenar y Exportar un Modelo](#7-entrenar-y-exportar-un-modelo)
8. [Despliegue con Docker](#8-despliegue-con-docker)
9. [Configuración](#9-configuración)
10. [Diseño Genérico y Extensibilidad](#10-diseño-genérico-y-extensibilidad)

---

## 1. Visión General

La API expone una interfaz REST genérica para **publicar y consumir modelos de ML de cribado preanestésico**. Está diseñada para desacoplarse completamente de cualquier versión concreta del modelo: toda la información sobre features, umbrales, estrategias de imputación y preprocesamiento viaja dentro del propio artefacto `.joblib`, no en el código de la API.

### Capacidades principales

| Capacidad | Detalle |
|-----------|---------|
| **Multi-modelo** | Carga simultánea de varios modelos (por versión clínica de target o tipo de algoritmo) |
| **Carga en caliente** | Nuevos modelos subidos via `POST /models/upload` sin reiniciar el servidor |
| **Imputación automática** | Features ausentes imputadas con la estrategia del entrenamiento (mediana o cero) |
| **Predicción individual y en lote** | Hasta `API_MAX_BATCH_SIZE` pacientes por llamada (default 100) |
| **Explicabilidad SHAP** | Atribución por feature activable por request (TreeExplainer / LinearExplainer / KernelExplainer) |
| **Agnóstico de framework** | Cualquier estimador sklearn-compatible funciona (RF, XGB, HistGB, LR, MLP…) |

---

## 2. Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                        api/                             │
│                                                         │
│  main.py ──── FastAPI app + lifespan (carga models/)   │
│     │                                                   │
│     ├── routers/                                        │
│     │     ├── health.py    GET /health                  │
│     │     ├── models.py    GET|POST|DELETE /models      │
│     │     └── predict.py   GET /predict/schema          │
│     │                      POST /predict                │
│     │                      POST /predict/batch          │
│     │                                                   │
│     ├── services/                                       │
│     │     ├── model_registry.py  ← carga .joblib       │
│     │     ├── feature_processor.py ← ENCODING_FIX_MAP  │
│     │     │                          imputación         │
│     │     │                          preprocessor       │
│     │     └── predictor.py  ← inferencia + SHAP        │
│     │                                                   │
│     ├── schemas/                                        │
│     │     ├── requests.py   PredictRequest, Batch…     │
│     │     └── responses.py  ModelInfo, Prediction…     │
│     │                                                   │
│     ├── core/config.py  ← Settings desde .env          │
│     └── scripts/                                        │
│           ├── train_and_save.py  ← CLI de entrenamiento│
│           └── create_demo_models.py  ← artefactos demo │
│                                                         │
│  models/  ← directorio de artefactos .joblib           │
└─────────────────────────────────────────────────────────┘
         ↑
    utils/  (pipeline de entrenamiento del proyecto)
```

### Flujo de una predicción

```
Request JSON
    │
    ▼
feature_processor.py
  1. Aplica ENCODING_FIX_MAP (corrige nombres con problemas UTF-8)
  2. Detecta features ausentes → imputa con estrategia del artefacto
  3. Si artifact["preprocessor"] presente → transforma (scaler, etc.)
    │
    ▼
predictor.py
  4. model.predict_proba(X)[:, 1]  →  probabilidad [0,1]
  5. predicted_class = int(prob >= optimal_threshold)
  6. risk_level = low | medium | high
  7. Si return_explanations=true → SHAP values (cacheado por model_id)
    │
    ▼
Response JSON
  { predicted_class, probability, risk_level, recommendation,
    threshold_used, version, explanations?, missing_features, warnings }
```

---

## 3. Formato del Artefacto de Modelo

Un artefacto es un **diccionario Python serializado con `joblib.dump()`**. Toda la lógica de inferencia reside en él.

```python
artifact = {
    # ── Obligatorios ──────────────────────────────────────────────────
    "model":              <estimador sklearn fitted>,     # cualquier modelo compatible
    "feature_columns":    ["Edad", "ASA", ...],          # lista ordenada de features
    "optimal_threshold":  0.38,                          # float en [0, 1]
    "version":            "target_b_clinicamente_relevante",
    "model_name":         "HistGradientBoosting",
    "model_type":         "tree",   # "tree" | "linear" | "ensemble" | "neural"

    # ── Recomendados ──────────────────────────────────────────────────
    "preprocessor":       <Pipeline/ColumnTransformer fitted | None>,
    "training_medians":   {"Edad": 51.0, "Peso": 73.8, ...},
    "imputation_strategies": {
        "fill_zero":   ["severity_score_critical_proc", "n_dx", ...],
        "fill_median": ["Edad", "Peso", "SpO2", ...],
    },
    "metrics": {
        "ROC-AUC":   0.91,
        "Recall":    0.85,
        "Precision": 0.88,
        "F1":        0.86,
        "F2":        0.85,
        "Threshold": 0.38,
    },
    "feature_importance": [            # lista de dicts ordenada por importancia desc
        {"feature": "ASA",             "importance": 0.243},
        {"feature": "flag_via_aerea",  "importance": 0.241},
        ...
    ],
    "created_at":          "2026-03-11",
    "description":         "HistGB entrenado sobre OPERA — versión B",
    "min_age":             18,
    "target_description":  "1 = requiere evaluación preanestésica, 0 = no indicada",
}

joblib.dump(artifact, "models/histgb_target_b.joblib")
```

> **Clave:** El nombre del archivo (sin `.joblib`) es el `model_id` que se usa en todos los endpoints.
> Ejemplo: `histgb_target_b.joblib` → `model_id = "histgb_target_b"`

### Estrategias de imputación

| Estrategia | Aplicar a | Valor |
|-----------|-----------|-------|
| `fill_zero` | Scores de severidad, flags, conteos, ordinales | `0` |
| `fill_median` | Variables fisiológicas (TA, FC, SpO2, Edad, IMC…) | mediana del conjunto de entrenamiento |

---

## 4. Instalación y Puesta en Marcha

### Requisitos

- Python 3.11+
- Las dependencias del proyecto raíz (`requirements.txt`) **y** las específicas de la API (`api/requirements.txt`)

### Instalación local

```bash
# Desde la raíz del proyecto
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### Iniciar el servidor

```bash
# Desarrollo (hot-reload)
uvicorn api.main:app --reload

# Producción
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Al arrancar, la API escanea automáticamente el directorio `models/` y carga todos los archivos `.joblib` que encuentre. Los logs de arranque muestran cada modelo cargado:

```
INFO  api.main – === Preanesthesia Screening API starting ===
INFO  api.main – Models directory: /app/models
INFO  api.services.model_registry –   ✓ Loaded 'histgb_target_b' (HistGradientBoosting / target_b_clinicamente_relevante)
INFO  api.services.model_registry –   ✓ Loaded 'rf_target_a' (Random Forest / target_a_sensible)
INFO  api.services.model_registry – ModelRegistry: 2 model(s) loaded.
```

---

## 5. Referencia de Endpoints

### `GET /health`

Verifica el estado del servicio.

**Response `200`**
```json
{
    "status": "ok",
    "models_loaded": 3,
    "version": "1.0.0",
    "shap_enabled": true
}
```

`status` es `"degraded"` si no hay ningún modelo cargado.

---

### `GET /models`

Lista todos los modelos disponibles con sus métricas y top features.

**Response `200`**
```json
{
    "models": [
        {
            "model_id": "histgb_target_b",
            "model_name": "HistGradientBoosting",
            "model_type": "tree",
            "version": "target_b_clinicamente_relevante",
            "description": "HistGB — target_b (demo)",
            "n_features": 20,
            "optimal_threshold": 0.38,
            "min_age": 18,
            "created_at": "2026-03-11",
            "metrics": {
                "roc_auc": 0.91,
                "recall": 0.85,
                "precision": 0.88,
                "f1": 0.86,
                "f2": null,
                "extra": {}
            },
            "top_features": [
                {"feature": "ASA",            "importance": 0.243},
                {"feature": "flag_via_aerea", "importance": 0.241}
            ]
        }
    ],
    "total": 1
}
```

---

### `GET /models/{model_id}`

Detalle completo de un modelo (incluye hasta 20 features por importancia).

```bash
curl http://localhost:8000/models/histgb_target_b
```

**Errores**

| Código | Causa |
|--------|-------|
| `404` | `model_id` no existe |

---

### `POST /models/upload`

Sube y registra un nuevo artefacto `.joblib` sin reiniciar el servidor.

**Parámetros de query**

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `model_id` | string | ✓ | Identificador único (letras, dígitos, guiones bajos) |
| `overwrite` | boolean | — | Permitir reemplazar un modelo existente (default `false`) |

```bash
curl -X POST "http://localhost:8000/models/upload?model_id=histgb_target_b" \
  -F "file=@models/histgb_target_b.joblib"
```

**Response `201`**
```json
{
    "model_id": "histgb_target_b",
    "message": "Model 'histgb_target_b' registered successfully.",
    "model_info": { ... }
}
```

**Errores**

| Código | Causa |
|--------|-------|
| `409` | El `model_id` ya existe y `overwrite=false` |
| `422` | Archivo vacío, no es `.joblib`, o falta alguna clave obligatoria del artefacto |

---

### `DELETE /models/{model_id}`

Elimina un modelo de memoria y opcionalmente del disco.

```bash
curl -X DELETE "http://localhost:8000/models/histgb_target_b?delete_file=true"
```

**Parámetros de query**

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `delete_file` | boolean | Si `true`, borra también el `.joblib` del directorio `models/` |

---

### `GET /predict/schema/{model_id}`

Devuelve la lista completa de features que espera el modelo, con su estrategia de imputación y la mediana de entrenamiento.

```bash
curl http://localhost:8000/predict/schema/rf_target_a
```

**Response `200`**
```json
{
    "model_id": "rf_target_a",
    "version": "target_a_sensible",
    "n_features": 20,
    "min_age": 18,
    "features": [
        {
            "name": "Edad",
            "imputation_strategy": "fill_median",
            "training_median": 51.0
        },
        {
            "name": "ASA",
            "imputation_strategy": "fill_median",
            "training_median": 3.0
        },
        {
            "name": "severity_score_critical_proc",
            "imputation_strategy": "fill_zero",
            "training_median": 0.0
        }
    ],
    "note": "Missing numeric features are imputed automatically. ..."
}
```

---

### `POST /predict`

Predicción para un único paciente.

**Request body**
```json
{
    "model_id": "rf_target_a",
    "patient_data": {
        "Edad": 72,
        "ASA": 4,
        "severity_score_critical_proc": 1,
        "flag_via_aerea_hist": 1,
        "n_dx": 8,
        "Peso": 85,
        "SpO2": 94,
        "Frecuencia Cardiaca": 98
    },
    "return_explanations": false
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `model_id` | string | ID del modelo a usar |
| `patient_data` | dict | Features del paciente. Las ausentes se imputan automáticamente |
| `return_explanations` | boolean | Incluir atribuciones SHAP por feature (default `false`) |

**Response `200` — Paciente alto riesgo**
```json
{
    "prediction": {
        "predicted_class": 1,
        "probability": 0.955,
        "risk_level": "high",
        "threshold_used": 0.35,
        "recommendation": "SCHEDULE PREANESTHESIA EVALUATION",
        "model_id": "rf_target_a",
        "version": "target_a_sensible",
        "explanations": null
    },
    "missing_features": ["Talla", "IMC", "Tensión Arterial Sistólica (mm/Hg)"],
    "warnings": []
}
```

**Response `200` — Paciente con `return_explanations: true`**
```json
{
    "prediction": {
        "predicted_class": 1,
        "probability": 0.82,
        "risk_level": "high",
        "threshold_used": 0.35,
        "recommendation": "SCHEDULE PREANESTHESIA EVALUATION",
        "model_id": "rf_target_a",
        "version": "target_a_sensible",
        "explanations": [
            {
                "feature": "ASA",
                "patient_value": 4.0,
                "shap_value": 0.312,
                "direction": "increases_risk"
            },
            {
                "feature": "flag_via_aerea_hist",
                "patient_value": 1.0,
                "shap_value": 0.289,
                "direction": "increases_risk"
            },
            {
                "feature": "Edad",
                "patient_value": 72.0,
                "shap_value": 0.145,
                "direction": "increases_risk"
            }
        ]
    },
    "missing_features": [],
    "warnings": []
}
```

**Response `200` — Paciente con edad < 18**
```json
{
    "prediction": { ... },
    "missing_features": [...],
    "warnings": [
        "Patient age (15) is below the model's minimum age (18). This model was trained on adult patients only; results may be unreliable."
    ]
}
```

**Niveles de riesgo**

| `risk_level` | Condición |
|-------------|-----------|
| `low` | `probability < threshold × 0.5` |
| `medium` | `threshold × 0.5 ≤ probability < threshold` o justo por encima del umbral |
| `high` | `probability ≥ threshold + (1 - threshold) × 0.5` |

---

### `POST /predict/batch`

Predicción para múltiples pacientes en una sola llamada.

**Request body**
```json
{
    "model_id": "rf_target_a",
    "patients": [
        {"Edad": 72, "ASA": 4, "severity_score_critical_proc": 1},
        {"Edad": 30, "ASA": 1, "severity_score_critical_proc": 0},
        {"Edad": 58, "ASA": 3, "Peso": 95}
    ],
    "return_explanations": false
}
```

**Response `200`**
```json
{
    "model_id": "rf_target_a",
    "total": 3,
    "successful": 3,
    "failed": 0,
    "results": [
        {
            "patient_index": 0,
            "prediction": {
                "predicted_class": 1,
                "probability": 0.955,
                "risk_level": "high",
                "threshold_used": 0.35,
                "recommendation": "SCHEDULE PREANESTHESIA EVALUATION",
                "model_id": "rf_target_a",
                "version": "target_a_sensible",
                "explanations": null
            },
            "error": null,
            "missing_features": ["Peso", "Talla", "IMC"],
            "warnings": []
        },
        {
            "patient_index": 1,
            "prediction": {
                "predicted_class": 0,
                "probability": 0.21,
                "risk_level": "low",
                "threshold_used": 0.35,
                "recommendation": "PREANESTHESIA EVALUATION NOT INDICATED",
                "model_id": "rf_target_a",
                "version": "target_a_sensible",
                "explanations": null
            },
            "error": null,
            "missing_features": ["Peso", "Talla", "IMC"],
            "warnings": []
        }
    ]
}
```

**Límites**

| Parámetro | Valor default | Configurable via |
|-----------|--------------|------------------|
| Tamaño máximo de lote | 100 | `API_MAX_BATCH_SIZE` |

---

## 6. Flujo Completo de Uso

### Paso 1 — Verificar que el servicio está activo

```bash
curl http://localhost:8000/health
# → { "status": "ok", "models_loaded": 3, ... }
```

### Paso 2 — Consultar modelos disponibles

```bash
curl http://localhost:8000/models
```

### Paso 3 — Obtener el schema de features del modelo elegido

```bash
curl http://localhost:8000/predict/schema/histgb_target_b
# → lista de 20 features con estrategia de imputación y mediana
```

### Paso 4 — Enviar una predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "histgb_target_b",
    "patient_data": {
      "Edad": 65,
      "ASA": 3,
      "severity_score_critical_proc": 1,
      "n_dx": 6,
      "SpO2": 95
    },
    "return_explanations": true
  }'
```

### Paso 5 — Subir un nuevo modelo sin reiniciar

```bash
curl -X POST "http://localhost:8000/models/upload?model_id=xgb_target_b" \
  -F "file=@models/xgb_target_b.joblib"
```

---

## 7. Entrenar y Exportar un Modelo

El script `api/scripts/train_and_save.py` integra el pipeline de entrenamiento existente (`utils/`) con la generación del artefacto en el formato correcto.

```bash
python -m api.scripts.train_and_save \
  --data     OPERA_COMPLETO.xlsx \
  --features variables_seleccionadas.csv \
  --version  target_b_clinicamente_relevante \
  --model    HistGradientBoosting \
  --output   models/histgb_target_b.joblib \
  --optimize          # activa búsqueda Optuna (30 trials por defecto)
  --n-trials 50       # número de trials Optuna
  --test-size 0.2
  --min-age   18
```

**Output del script**

```
============================================================
  Model: HistGradientBoosting  |  Version: target_b_clinicamente_relevante
  Features: 80
  Threshold: 0.382
  ROC-AUC:  0.8934
  Recall:   0.8621
  F2:       0.8497
  Artifact: models/histgb_target_b.joblib
============================================================
```

Tras ejecutarse, el modelo queda disponible inmediatamente subiéndolo via la API o reiniciando el servidor (si está en el directorio `models/`).

### Construcción manual del artefacto

Si se entrena desde un notebook, el artefacto se ensambla así:

```python
import joblib

artifact = {
    # Obligatorios
    "model":             modelo_entrenado,
    "feature_columns":   selected_features,       # list[str]
    "optimal_threshold": optimal_threshold,        # float, resultado de find_optimal_threshold()
    "version":           "target_b_clinicamente_relevante",
    "model_name":        "XGBoost",
    "model_type":        "tree",

    # Recomendados
    "preprocessor":      preprocessor_fitted,      # sklearn Pipeline o None
    "training_medians":  X_train.median().to_dict(),
    "imputation_strategies": get_imputation_strategies(selected_features),
    "metrics":           metrics_dict,             # de compute_classification_metrics()
    "feature_importance": [{"feature": c, "importance": float(v)} ...],
    "created_at":        "2026-03-11",
    "description":       "XGBoost optimizado con Optuna — 80 features — target_b",
    "min_age":           18,
    "target_description": "1 = requiere evaluación preanestésica",
}

joblib.dump(artifact, "models/xgb_target_b.joblib")
```

---

## 8. Despliegue con Docker

### Build y arranque

```bash
docker-compose up --build
```

La API queda expuesta en `http://localhost:8000`.

### `docker-compose.yml` — variables clave

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models   # artefactos persistidos fuera del contenedor
    environment:
      - API_DEBUG=false
      - API_MAX_BATCH_SIZE=100
      - API_ENABLE_SHAP=true
```

El volumen `./models:/app/models` es crítico: permite agregar nuevos modelos en producción sin reconstruir la imagen.

### Dockerfile — estrategia multi-stage

La imagen sigue un patrón **builder → runtime** que reduce el tamaño final:

1. **builder**: instala dependencias de compilación y paquetes Python
2. **runtime**: copia solo los paquetes compilados + código fuente

```dockerfile
ENV PYTHONPATH=/app   # hace importable utils/ desde api/
```

El proceso corre como usuario no-root (`apiuser`) por seguridad.

### Health check integrado

Docker verifica el estado del contenedor cada 30 segundos:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  retries: 3
```

---

## 9. Configuración

Todas las variables se configuran mediante **variables de entorno** con prefijo `API_`, o en un archivo `.env` en la raíz del proyecto.

| Variable | Default | Descripción |
|----------|---------|-------------|
| `API_MODELS_DIR` | `models` | Directorio de artefactos `.joblib` |
| `API_MAX_BATCH_SIZE` | `100` | Máximo de pacientes por request de batch |
| `API_ENABLE_SHAP` | `true` | Habilitar/deshabilitar SHAP globalmente |
| `API_SHAP_BACKGROUND_SAMPLES` | `100` | Filas para KernelExplainer (modelos lineales/neurales) |
| `API_DEBUG` | `false` | Logging detallado y modo debug de FastAPI |
| `API_HOST` | `0.0.0.0` | Dirección de escucha (solo CLI local) |
| `API_PORT` | `8000` | Puerto de escucha (solo CLI local) |

Copiar el template de configuración:

```bash
cp .env.example .env
```

---

## 10. Diseño Genérico y Extensibilidad

### Por qué es genérica

La API **no tiene hardcodeado ningún nombre de feature, umbral, ni tipo de modelo**. Toda esa información reside en el artefacto:

```
artifact["feature_columns"]          → qué features esperar
artifact["optimal_threshold"]        → dónde cortar la probabilidad
artifact["imputation_strategies"]    → cómo imputar missing values
artifact["preprocessor"]             → cómo transformar antes de inferir
artifact["model_type"]               → qué tipo de SHAP Explainer usar
```

Cambiar el modelo clínico, añadir features nuevas o modificar el threshold **no requiere tocar el código de la API**: basta re-entrenar y subir el nuevo `.joblib`.

### Añadir un nuevo algoritmo

Cualquier estimador que implemente `predict_proba(X)` de scikit-learn es compatible de forma inmediata (LightGBM, CatBoost, redes neuronales con wrapper sklearn, etc.).

Para SHAP, el `model_type` del artefacto determina el explainer:

| `model_type` | SHAP Explainer usado |
|-------------|----------------------|
| `"tree"` | `TreeExplainer` (rápido, exacto) |
| `"linear"` | `LinearExplainer` |
| `"ensemble"` o `"neural"` | `KernelExplainer` (lento, aproximado) |

### Añadir una nueva versión clínica de target

1. Entrenar el modelo con la nueva definición de target en `utils/version_config.py`
2. Exportar el artefacto con `model_type`, `version` y `optimal_threshold` correspondientes
3. Subirlo via `POST /models/upload?model_id=<nuevo_id>`

No hay ningún cambio de código necesario.

### Referencia de la documentación interactiva

Con el servidor corriendo, acceder a:

- **Swagger UI:** `http://localhost:8000/docs` — interfaz de prueba interactiva
- **ReDoc:** `http://localhost:8000/redoc` — documentación legible

---

## Estructura de Archivos de la API

```
api/
├── main.py                        # FastAPI app, CORS, lifespan
├── requirements.txt               # Dependencias específicas de la API
├── core/
│   └── config.py                  # Settings (pydantic-settings, env vars)
├── schemas/
│   ├── requests.py                # PredictRequest, BatchPredictRequest
│   └── responses.py               # ModelInfo, Prediction, PredictResponse…
├── services/
│   ├── model_registry.py          # Carga .joblib, CRUD de modelos en memoria
│   ├── feature_processor.py       # ENCODING_FIX_MAP, imputación, preprocessor
│   └── predictor.py               # Inferencia, SHAP (con caché de explainers)
├── routers/
│   ├── health.py
│   ├── models.py
│   └── predict.py
└── scripts/
    ├── train_and_save.py          # CLI: entrena modelo y guarda artefacto
    └── create_demo_models.py      # Genera 3 artefactos de demo para pruebas
```
