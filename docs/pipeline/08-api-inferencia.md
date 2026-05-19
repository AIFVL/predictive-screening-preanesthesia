# 08 — API REST de Inferencia (FastAPI)

> **Proyecto:** Screening predictivo de valoración preanestésica — Fundación Valle del Lili
> **Etapa:** 8 de 8 — Servicio de inferencia en producción

---

## Outputs / Artefactos

| Artefacto | Ruta | Descripción |
|---|---|---|
| Punto de entrada | `api/main.py` | App FastAPI, lifespan, middlewares |
| Configuración | `api/core/config.py` | Settings, variables de entorno, aliases de target |
| Routers | `api/routers/` | Módulos `health`, `targets`, `models`, `predict` |
| Schemas Pydantic | `api/schemas/` | Contratos de request y response |
| Registro de modelos | `api/domain/registry.py` | Descubrimiento y caché LRU de modelos |
| Manifest | `api/domain/manifest.py` | Dataclass que representa un `*_manifest.json` |
| Servicio de predicción | `api/services/predictor.py` | `predict_one` / `predict_batch` |
| Preprocesamiento | `api/services/preprocessor.py` | Dict → DataFrame → imputación |
| Dependencias Python | `api/requirements.txt` | FastAPI, Uvicorn, scikit-learn, XGBoost, etc. |
| Imagen Docker | `api/Dockerfile` | Imagen multi-stage basada en `python:3.11-slim` |
| Variables de entorno | `api/.env.example` | Plantilla de configuración |
| Compose | `docker-compose.yaml` | Servicio `api` + Airflow + PostgreSQL |
| Tests E2E | `tests/api/` | 23 casos de prueba (salud, descubrimiento, predicción, lote, latencia, robustez) |

---

## 1. Propósito

La API expone los modelos entrenados en el pipeline preanestésico como un servicio REST, permitiendo que sistemas externos obtengan predicciones de riesgo para un paciente en tiempo real sin necesidad de conocer los detalles del entrenamiento. Su audiencia principal comprende los sistemas de información hospitalaria (HIS/HCE) que integran el screening automatizado en el flujo de valoración preanestésica, las interfaces clínicas que presentan al anestesiólogo el nivel de riesgo del paciente antes de la consulta, y las herramientas de análisis que consumen predicciones en modo batch.

El principio de diseño fundamental es el desacoplamiento entre entrenamiento e inferencia: junto al archivo `.joblib`, el pipeline produce un `<algoritmo>_manifest.json` con el contrato completo del modelo (features, dtypes, threshold, calibración, imputación). La API consume esos manifests directamente, sin depender del código del pipeline. Cualquier modelo nuevo entrenado queda disponible para inferencia sin modificar el código de la API.

---

## 2. Arquitectura

### Estructura de la aplicación

```
api/
├── main.py                  # create_app(), lifespan (descubrimiento de modelos al arrancar)
├── core/
│   ├── config.py            # Settings (dataclass frozen), TARGET_ALIASES, get_settings()
│   ├── logging.py           # Logging estructurado
│   └── sklearn_compat.py    # Parches de compatibilidad de versiones sklearn
├── domain/
│   ├── manifest.py          # ModelManifest (dataclass frozen), load_manifest()
│   └── registry.py          # ModelRegistry: descubrimiento, caché LRU, list/get
├── schemas/
│   ├── common.py            # ApiResponse, success_response(), error_response()
│   ├── models.py            # ModelSummary, ModelDetail, ModelSchema, FeatureSpec
│   └── predict.py           # PredictResponse, BatchPredictResponse, validadores dinámicos
├── services/
│   ├── predictor.py         # predict_one(), predict_batch(), _risk_level()
│   └── preprocessor.py      # features_dict_to_dataframe(), apply_imputation(), preprocess()
├── routers/
│   ├── health.py            # GET /health, GET /ready
│   ├── targets.py           # GET /targets
│   ├── models.py            # GET /models, GET /models/{target}/{algorithm}, GET /models/{target}/{algorithm}/schema
│   └── predict.py           # POST /models/{target}/{algorithm}/predict, POST .../predict/batch
├── Dockerfile
├── requirements.txt
└── .env.example
```

### Targets disponibles (slugs públicos de la API)

La configuración en `api/core/config.py` define los dos targets en producción:

| Slug API | `target_version` interno | Display name | Recomendado |
|---|---|---|---|
| `general_risk` | `target_d_v2_hosp` | Riesgo general | No |
| `hospitalization_risk` | `target_f_predictibilidad_maxima` | Riesgo de hospitalización / UCI | **Sí** |

### Envoltura de respuesta

Todas las respuestas siguen el mismo envelope `ApiResponse`:

```json
{
  "success": true,
  "data": { ... },
  "errors": null,
  "meta": {
    "request_id": "<uuid_hex>",
    "model_id": "<model_id_o_null>"
  }
}
```

En caso de error de negocio (input inválido, batch vacío, etc.) el campo `success` es `false`, `data` es `null` y `errors` contiene una lista de `ErrorDetail` con campos `code`, `message` y `field`.

---

## 3. Carga de modelos

### Descubrimiento al arrancar (lifespan)

Al iniciar la app, el `lifespan` de FastAPI instancia `ModelRegistry` y llama a `registry.discover()`. Este método:

1. Itera sobre los subdirectorios de `MODELS_DIR` (por defecto `output/v2/models/`).
2. Para cada subdirectorio cuyo nombre coincida con un `target_version` registrado en `TARGET_ALIASES`, escanea todos los archivos `*_manifest.json`.
3. Carga el manifest con `load_manifest()` y valida que el `.joblib` referenciado en `model_filename` exista en disco.
4. Registra el modelo bajo la clave `(target_slug, algorithm)`.

Si ningún modelo queda registrado, la app arranca con una advertencia pero sigue respondiendo (`/health` retorna ok; `/ready` retorna `no_models`).

### Archivos requeridos por modelo

Para que un modelo sea servible necesita dos archivos en `MODELS_DIR/{target_version}/`:

| Archivo | Ejemplo | Descripción |
|---|---|---|
| `{algorithm}_manifest.json` | `xgboost_manifest.json` | Contrato del modelo: features, dtypes, medianas, threshold, calibración, imputación, métricas |
| `{algorithm}_model.joblib` | `xgboost_model.joblib` | Estimador serializado (sklearn / XGBoost / joblib) |

### Caché LRU de modelos en memoria

Los modelos `.joblib` se cargan de forma **lazy** (al primer request que los usa) y se mantienen en una caché LRU con capacidad configurable (`MODEL_CACHE_SIZE`, por defecto 8). Cuando la caché alcanza su capacidad máxima, el modelo menos recientemente utilizado es desalojado. La caché está protegida con `threading.Lock` para garantizar seguridad en entornos concurrentes.

### Preprocesamiento de features en inferencia

El servicio `preprocessor.preprocess()` aplica el siguiente pipeline antes de llamar a `model.predict_proba()`:

1. Convierte el dict (o lista de dicts) de features a `pd.DataFrame`.
2. Reordena las columnas según `manifest.feature_names` (las columnas ausentes quedan como `NaN`).
3. Aplica imputación según `manifest.imputation`:
   - `fill_constant` (valor `-1` por defecto): rellena `NaN` con el valor centinela.
   - `fill_median`: rellena `NaN` con la mediana de entrenamiento almacenada en el manifest.

Esto significa que **todos los campos son opcionales en el request**: si un campo no se envía, el modelo recibe el valor de imputación configurado en su manifest.

---

## 4. Endpoints documentados

### 4.1 `GET /health`

**Descripción:** Liveness probe. Retorna `ok` sin consultar el estado de los modelos.

**Response:**
```json
{
  "success": true,
  "data": { "status": "ok" },
  "errors": null,
  "meta": { "request_id": "a1b2c3...", "model_id": null }
}
```

**curl:**
```bash
curl http://localhost:8000/health
```

---

### 4.2 `GET /ready`

**Descripción:** Readiness probe. Retorna `ready` si hay al menos un modelo registrado, o `no_models` si la API arrancó sin modelos.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "ready",
    "n_models": 8,
    "cache_loaded": 2
  },
  "errors": null,
  "meta": { "request_id": "a1b2c3...", "model_id": null }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `status` | `string` | `"ready"` o `"no_models"` |
| `n_models` | `int` | Modelos registrados desde disco |
| `cache_loaded` | `int` | Modelos actualmente cargados en memoria |

**curl:**
```bash
curl http://localhost:8000/ready
```

---

### 4.3 `GET /targets`

**Descripción:** Lista los slugs de target disponibles. Útil para construir el primer selector en una interfaz clínica.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "slug": "hospitalization_risk",
      "display_name": "Riesgo de hospitalización / UCI",
      "description": "Target específico de outcomes graves...",
      "recommended": true,
      "n_models": 4
    },
    {
      "slug": "general_risk",
      "display_name": "Riesgo general",
      "description": "Target amplio que cubre múltiples flags clínicos preoperatorios...",
      "recommended": false,
      "n_models": 4
    }
  ],
  "errors": null,
  "meta": { "request_id": "...", "model_id": null }
}
```

**curl:**
```bash
curl http://localhost:8000/targets
```

---

### 4.4 `GET /models`

**Descripción:** Lista plana de todos los modelos servidos con su metadata resumida.

**Response (item de ejemplo):**
```json
{
  "success": true,
  "data": [
    {
      "target": "hospitalization_risk",
      "target_display_name": "Riesgo de hospitalización / UCI",
      "algorithm": "xgboost",
      "model_id": "target_f_predictibilidad_maxima__xgboost",
      "calibrated": true,
      "calibration_method": "isotonic",
      "threshold": 0.14,
      "performance": {
        "roc_auc": 0.8608,
        "pr_auc": 0.7039,
        "f2": 0.6769,
        "recall": 0.8528,
        "precision": 0.3708
      },
      "warnings": [],
      "recommended": true
    }
  ],
  "errors": null,
  "meta": { "request_id": "...", "model_id": null }
}
```

**curl:**
```bash
curl http://localhost:8000/models
```

---

### 4.5 `GET /models/{target}/{algorithm}`

**Descripción:** Metadata completa de un modelo específico.

**Parámetros de ruta:**

| Parámetro | Tipo | Ejemplo |
|---|---|---|
| `target` | `string` | `hospitalization_risk` |
| `algorithm` | `string` | `xgboost` |

**Response (campos principales):**

| Campo | Tipo | Descripción |
|---|---|---|
| `model_id` | `string` | Identificador único del modelo |
| `target` | `string` | Slug del target |
| `target_display_name` | `string` | Nombre legible |
| `algorithm` | `string` | Algoritmo de ML |
| `calibrated` | `bool` | Si el modelo tiene calibración de probabilidades |
| `calibration` | `object` | `{"method": "isotonic", "cv": 5, "fallback_reason": null}` |
| `threshold` | `float` | Umbral de decisión optimizado (típicamente 0.13–0.18) |
| `threshold_metric` | `string` | Métrica usada para optimizar el threshold (`"f2"`) |
| `prevalence` | `object` | Prevalencia en train/test |
| `performance` | `object` | Métricas de evaluación en test |
| `warnings` | `list[string]` | Advertencias asociadas al modelo |
| `n_features` | `int` | Número de features del modelo |
| `created_at` | `string\|null` | Timestamp ISO 8601 de creación |

**curl:**
```bash
curl http://localhost:8000/models/hospitalization_risk/xgboost
```

**Errores:**
- `404` si `target` o `algorithm` no están registrados.

---

### 4.6 `GET /models/{target}/{algorithm}/schema`

**Descripción:** Schema accionable del modelo: lista de features con nombre, tipo y mediana de entrenamiento. Permite a un frontend renderizar el formulario de entrada con valores por defecto.

**Parámetros de ruta:** igual que `4.5`.

**Response (campos principales):**

| Campo | Tipo | Descripción |
|---|---|---|
| `model_id` | `string` | Identificador del modelo |
| `target` | `string` | Slug del target |
| `algorithm` | `string` | Algoritmo |
| `features` | `list[FeatureSpec]` | Lista ordenada de features (ver tabla abajo) |
| `threshold` | `float` | Umbral de decisión |
| `threshold_metric` | `string` | Métrica de optimización del threshold |
| `prevalence` | `object` | Prevalencia en train/test |
| `calibrated` | `bool` | Si está calibrado |
| `calibration_method` | `string\|null` | Método de calibración |
| `imputation` | `object` | `{"strategy": "fill_constant", "value": -1}` |
| `warnings` | `list[string]` | Advertencias |

**Estructura de `FeatureSpec`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | `string` | Nombre exacto de la feature (tal como se envía en el request) |
| `dtype` | `string` | Tipo de dato: `"float64"`, `"int64"` o `"bool"` |
| `required` | `bool` | Siempre `false` — todas las features son opcionales (el preprocessor imputa) |
| `median` | `float\|null` | Mediana del conjunto de entrenamiento (útil como valor por defecto en UI) |

**curl:**
```bash
curl http://localhost:8000/models/hospitalization_risk/xgboost/schema
```

---

### 4.7 `POST /models/{target}/{algorithm}/predict`

**Descripción:** Predicción individual. Recibe las features de un paciente y retorna la clase predicha, la probabilidad calibrada y el nivel de riesgo.

**Parámetros de ruta:** igual que `4.5`.

**Request body:**

```json
{
  "features": {
    "Edad": 35,
    "Peso (Kg)": 68,
    "Talla (cm)": 170,
    "IMC": 23.5,
    "Tipo de anestesia propuesta_local": 1,
    "Tipo de anestesia propuesta_general": 0,
    "score_proc_low_severity": 1,
    "score_proc_high_severity": 0,
    "score_proc_critical": 0,
    "score_dx_critical": 0,
    "score_dx_high_severity": 0,
    "Examen_Hemoglobina(g/dl)": 14.2,
    "Antecedente endocrinológicos_negativo": 1,
    "Antecedente renales_negativo": 1,
    "Antecedente hematológicos _negativo": 1
  }
}
```

El objeto `features` acepta cualquier subconjunto de las features del modelo, cuya lista completa puede obtenerse mediante `/schema`. Las features no incluidas en el request se imputan automáticamente. No se permiten campos desconocidos (`extra="forbid"`).

**Response:**

```json
{
  "success": true,
  "data": {
    "predicted_class": 0,
    "probability": 0.0821,
    "threshold": 0.14,
    "risk_level": "low",
    "calibrated": true,
    "prevalence_train": 0.1943,
    "warnings": []
  },
  "errors": null,
  "meta": {
    "request_id": "f3a2b1...",
    "model_id": "target_f_predictibilidad_maxima__xgboost"
  }
}
```

**Campos de respuesta:**

| Campo | Tipo | Descripción |
|---|---|---|
| `predicted_class` | `int` | `0` (no requiere valoración adicional) o `1` (requiere valoración) |
| `probability` | `float` | Probabilidad calibrada de pertenecer a la clase positiva (rango 0–1) |
| `threshold` | `float` | Umbral aplicado para producir `predicted_class` |
| `risk_level` | `string` | Categoría de riesgo: `"low"`, `"moderate"`, `"elevated"` o `"high"` |
| `calibrated` | `bool` | Si la probabilidad proviene de un modelo calibrado |
| `prevalence_train` | `float\|null` | Prevalencia en entrenamiento (referencia para interpretar la probabilidad) |
| `warnings` | `list[string]` | Advertencias del modelo (p. ej. `"model_not_calibrated"`) |

**Lógica de `risk_level`:**

| Condición | `risk_level` |
|---|---|
| `probability >= threshold * 1.5` | `"high"` |
| `threshold <= probability < threshold * 1.5` | `"elevated"` |
| `threshold * 0.5 <= probability < threshold` | `"moderate"` |
| `probability < threshold * 0.5` | `"low"` |

**curl (paciente de bajo riesgo):**
```bash
curl -X POST http://localhost:8000/models/hospitalization_risk/xgboost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Edad": 35,
      "Peso (Kg)": 68,
      "Talla (cm)": 170,
      "IMC": 23.5,
      "Tipo de anestesia propuesta_local": 1,
      "Tipo de anestesia propuesta_general": 0,
      "score_proc_low_severity": 1,
      "score_proc_high_severity": 0,
      "score_proc_critical": 0,
      "Examen_Hemoglobina(g/dl)": 14.2
    }
  }'
```

**curl (sin features — imputa todo con valor centinela):**
```bash
curl -X POST http://localhost:8000/models/hospitalization_risk/xgboost/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {}}'
```

**Errores de negocio (HTTP 200, `success: false`):**

| `code` | Causa |
|---|---|
| `invalid_input` | Feature desconocida, tipo incorrecto o falta la clave `features` |

**Errores HTTP:**

| Código | Causa |
|---|---|
| `404` | `target` o `algorithm` no registrado |
| `500` | Error interno al cargar el modelo o ejecutar `predict_proba` |

---

### 4.8 `POST /models/{target}/{algorithm}/predict/batch`

**Descripción:** Predicción en lote. Recibe una lista de pacientes y retorna una predicción por cada uno, en el mismo orden.

**Parámetros de ruta:** igual que `4.5`.

**Request body:**

```json
{
  "items": [
    {
      "Edad": 35,
      "Peso (Kg)": 68,
      "Tipo de anestesia propuesta_local": 1
    },
    {
      "Edad": 78,
      "IMC": 33.8,
      "score_proc_critical": 1,
      "Examen_Hemoglobina(g/dl)": 9.1
    }
  ]
}
```

Cada elemento de `items` sigue el mismo schema que el campo `features` del endpoint individual. El batch máximo es configurable (`MAX_BATCH_SIZE`, por defecto 100 items).

**Response:**

```json
{
  "success": true,
  "data": {
    "n": 2,
    "predictions": [
      {
        "predicted_class": 0,
        "probability": 0.0821,
        "threshold": 0.14,
        "risk_level": "low",
        "calibrated": true,
        "prevalence_train": 0.1943,
        "warnings": []
      },
      {
        "predicted_class": 1,
        "probability": 0.7312,
        "threshold": 0.14,
        "risk_level": "high",
        "calibrated": true,
        "prevalence_train": 0.1943,
        "warnings": []
      }
    ]
  },
  "errors": null,
  "meta": { "request_id": "...", "model_id": "target_f_predictibilidad_maxima__xgboost" }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `n` | `int` | Número de predicciones (igual al número de items enviados) |
| `predictions` | `list[PredictResponse]` | Lista de predicciones en el mismo orden que `items` |

**curl:**
```bash
curl -X POST http://localhost:8000/models/hospitalization_risk/xgboost/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"Edad": 35, "Peso (Kg)": 68},
      {"Edad": 78, "score_proc_critical": 1}
    ]
  }'
```

**Errores de negocio (HTTP 200, `success: false`):**

| `code` | Causa |
|---|---|
| `invalid_input` | Feature desconocida o tipo incorrecto en algún item |
| `empty_batch` | `items` es una lista vacía |
| `batch_too_large` | `items` supera `MAX_BATCH_SIZE` |

---

## 5. Configuración y variables de entorno

La aplicación se configura exclusivamente a través de variables de entorno. La plantilla completa se encuentra en `api/.env.example`.

| Variable | Por defecto | Descripción |
|---|---|---|
| `PROJECT_ROOT` | `.` (directorio de trabajo) | Raíz del proyecto. Se usa para resolver `MODELS_DIR` si no se especifica. |
| `MODELS_DIR` | `$PROJECT_ROOT/output/v2/models` | Directorio raíz de modelos. La API escanea subdirectorios en busca de `*_manifest.json`. |
| `LOG_LEVEL` | `INFO` | Nivel de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `CORS_ORIGINS` | `*` | Orígenes permitidos para CORS, separados por coma. Ejemplo: `http://localhost:3000,https://mi-dominio.com`. |
| `MAX_BATCH_SIZE` | `100` | Tamaño máximo de batch en `/predict/batch`. |
| `MODEL_CACHE_SIZE` | `8` | Número máximo de modelos cargados simultáneamente en memoria (LRU). |

Ejemplo de `.env.example`:
```
API_PORT=8000
API_LOG_LEVEL=INFO
API_CORS_ORIGINS=http://localhost:3000,https://midominio.com
API_MAX_BATCH_SIZE=100
API_MODEL_CACHE_SIZE=8
```

> Nota: en el `docker-compose.yaml` las variables se pasan sin el prefijo `API_` (p. ej. `LOG_LEVEL`, `CORS_ORIGINS`). El prefijo en `.env.example` es solo una convención para facilitar la carga con `--env-file`.

---

## 6. Despliegue con Docker

### Imagen (`api/Dockerfile`)

La imagen emplea un build multi-stage basado en `python:3.11-slim`. El stage `builder` instala las dependencias de `api/requirements.txt` en el directorio del usuario; el stage `runtime` copia únicamente los paquetes instalados y el código de `api/`, ejecuta el proceso como usuario no-root (`api`, UID 1000), expone el puerto `8000` e incorpora un healthcheck interno con `curl -fs http://localhost:8000/health`.

Comando de inicio del contenedor:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Servicio `api` en `docker-compose.yaml`

El compose levanta el servicio `api` junto con Airflow y PostgreSQL. El directorio `output/` se monta como volumen de solo lectura para que la API acceda a los modelos generados por el pipeline:

```yaml
api:
  build:
    context: .
    dockerfile: api/Dockerfile
  image: preanesthesia-api:latest
  container_name: preanesthesia-api
  ports:
    - "${API_PORT:-8000}:8000"
  environment:
    MODELS_DIR: /app/output/v2/models
    LOG_LEVEL: "${API_LOG_LEVEL:-INFO}"
    CORS_ORIGINS: "${API_CORS_ORIGINS:-*}"
    MAX_BATCH_SIZE: "${API_MAX_BATCH_SIZE:-100}"
    MODEL_CACHE_SIZE: "${API_MODEL_CACHE_SIZE:-8}"
  volumes:
    - ./output:/app/output:ro
  healthcheck:
    test: ["CMD", "curl", "-fs", "http://localhost:8000/health"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 15s
  restart: unless-stopped
```

### Comandos de despliegue

```bash
# Copiar y ajustar la configuración
cp api/.env.example .env
# Editar .env con los valores de producción

# Construir y levantar solo la API (sin Airflow)
docker compose up -d api

# Levantar el stack completo (Airflow + API)
docker compose up -d

# Verificar que el servicio esté sano
docker compose ps
docker compose logs api -f

# Healthcheck manual
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Detener sin borrar volúmenes
docker compose down
```

### Despliegue sin Docker (desarrollo local)

```bash
# Instalar dependencias
pip install -r api/requirements.txt

# Configurar variables de entorno
export PROJECT_ROOT=$(pwd)
export MODELS_DIR=$(pwd)/output/v2/models
export LOG_LEVEL=INFO

# Iniciar el servidor
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentación interactiva (Swagger UI) queda disponible en `http://localhost:8000/docs`.

---

## 7. Tests

Los tests de extremo a extremo se encuentran en `tests/api/` y utilizan `pytest` con `fastapi.testclient.TestClient`. El fixture `conftest.py` configura automáticamente `PROJECT_ROOT` y `MODELS_DIR` apuntando a `output/v2/models/`.

### Archivos de test

| Archivo | Tests | Qué cubren |
|---|---|---|
| `test_e2e_health.py` | TC01, TC02 | `/health` y `/ready` |
| `test_e2e_discovery.py` | TC03–TC06 | `/targets`, `/models`, `/models/{t}/{a}`, `/models/{t}/{a}/schema` |
| `test_e2e_predict_happy.py` | TC07–TC10 | Predicción individual: paciente vacío, bajo riesgo, alto riesgo, contrato de metadatos |
| `test_e2e_predict_validation.py` | TC11–TC15 | Validaciones: falta `features`, feature desconocida, dtype incorrecto, target/algoritmo inexistente |
| `test_e2e_predict_batch.py` | TC16–TC19 | Batch: happy path, batch vacío, batch demasiado grande, item con feature inválida |
| `test_e2e_predict_latency.py` | TC20–TC21 | Latencia mediana ≤ 1500 ms (individual) y ≤ 5000 ms (batch de 50) |
| `test_e2e_predict_robustness.py` | TC22–TC23 | Fallo de modelo → HTTP 500; recuperación tras fallo |

### Cómo ejecutar los tests

```bash
# Desde la raíz del proyecto
pytest tests/api/ -v

# Solo tests de predicción
pytest tests/api/test_e2e_predict_happy.py tests/api/test_e2e_predict_validation.py -v

# Excluir tests de latencia (más lentos)
pytest tests/api/ -v --ignore=tests/api/test_e2e_predict_latency.py

# Con variables de entorno explícitas
MODELS_DIR=output/v2/models pytest tests/api/ -v
```

> **Requisito:** los tests asumen que `output/v2/models/` contiene al menos el modelo `hospitalization_risk/xgboost` (manifest + joblib). Si el modelo no está disponible, los tests que lo requieren se omiten automáticamente mediante `pytest.skip`.

### Fixture `conftest.py` — datos de ejemplo

El conftest define dos fixtures de features representativas usadas en múltiples tests:

**`low_risk_features`** — paciente joven, bajo peso quirúrgico, anestesia local:
```json
{
  "Edad": 35, "Peso (Kg)": 68, "Talla (cm)": 170, "IMC": 23.5,
  "Tipo de anestesia propuesta_local": 1, "Tipo de anestesia propuesta_general": 0,
  "score_proc_low_severity": 1, "score_proc_high_severity": 0, "score_proc_critical": 0,
  "score_dx_critical": 0, "score_dx_high_severity": 0,
  "Examen_Hemoglobina(g/dl)": 14.2,
  "Antecedente endocrinológicos_negativo": 1, "Antecedente renales_negativo": 1,
  "Antecedente hematológicos _negativo": 1
}
```

**`high_risk_features`** — paciente mayor, obeso, cirugía crítica, anemia:
```json
{
  "Edad": 78, "Peso (Kg)": 92, "Talla (cm)": 165, "IMC": 33.8,
  "Tipo de anestesia propuesta_general": 1, "Tipo de anestesia propuesta_local": 0,
  "score_proc_critical": 1, "score_proc_high_severity": 1, "score_proc_low_severity": 0,
  "score_dx_critical": 1, "score_dx_high_severity": 1,
  "Examen_Hemoglobina(g/dl)": 9.1
}
```
