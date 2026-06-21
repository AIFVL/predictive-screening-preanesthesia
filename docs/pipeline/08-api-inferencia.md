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
| Servicio de predicción | `api/services/predictor.py` | `predict_one()` |
| Preprocesamiento clínico | `api/services/clinical_preprocessor.py` | Variables clínicas crudas → limpieza + enriquecimiento → espacio de features |
| Imputación de features | `api/services/preprocessor.py` | Reindexado e imputación según el manifest |
| Explicabilidad | `api/services/explainer.py` | Contribuciones SHAP por caso (`explain_one()`) |
| Dependencias Python | `api/requirements.txt` | FastAPI, Uvicorn, scikit-learn, XGBoost, transformers/torch (BART-MNLI), NLTK, etc. |
| Imagen Docker | `api/Dockerfile` | Imagen multi-stage basada en `python:3.11-slim` (copia `api/`, `src/`, `config/` + stopwords NLTK) |
| Variables de entorno | `api/.env.example` | Plantilla de configuración |
| Compose | `docker-compose.yaml` | Servicio `api` + Airflow + PostgreSQL |
| Tests E2E | `tests/api/` | 6 casos de prueba (salud y descubrimiento de modelos) |

---

## 1. Propósito

La API expone los modelos entrenados en el pipeline preanestésico como un servicio REST, permitiendo que sistemas externos obtengan predicciones de riesgo para un paciente en tiempo real sin necesidad de conocer los detalles del entrenamiento. Su audiencia principal comprende los sistemas de información hospitalaria (HIS/HCE) que integran el screening automatizado en el flujo de valoración preanestésica, las interfaces clínicas que presentan al anestesiólogo el nivel de riesgo del paciente antes de la consulta, y las herramientas de análisis que consumen predicciones en modo batch.

El principio de diseño fundamental es el desacoplamiento entre entrenamiento e inferencia: junto al archivo `.joblib`, el pipeline produce un `<algoritmo>_manifest.json` con el contrato completo del modelo (features, dtypes, threshold, calibración, imputación, y además `raw_input_schema` + `raw_input_example`). La API consume esos manifests directamente, sin depender del estado del pipeline. Cualquier modelo nuevo entrenado queda disponible para inferencia sin modificar el código de la API.

**La API recibe variables clínicas crudas, no features procesadas.** El cliente envía los datos del paciente con los nombres de columna originales del dataset (español, con tildes y espacios; p. ej. `"Tensión Arterial Sistólica (mm/Hg)"`, `"Antecedentes cardiovasculares"`), exactamente como los describe `raw_input_schema` en el manifest. La API aplica internamente el **mismo pipeline de limpieza y enriquecimiento** que se usó en entrenamiento (limpieza determinista vía `clean_preop()` + codificación ICD y severidad clínica con BART-MNLI vía `enrich_preop()`), reutilizando el código de `src/cleaning/`. Esto garantiza paridad train/serve: el cliente no necesita conocer las ~64 features internas del modelo ni cómo se derivan.

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
│   ├── models.py            # ModelDetail, ModelSchema, FeatureSpec
│   └── predict.py           # PredictResponse, ExplainResponse, ShapContributionSchema
├── services/
│   ├── predictor.py         # predict_one(), _risk_level()
│   ├── clinical_preprocessor.py  # preprocess_raw(): crudo → clean_preop + enrich_preop → features
│   ├── preprocessor.py      # apply_imputation() (reindexa al manifest e imputa)
│   └── explainer.py         # explain_one(): contribuciones SHAP top-N por caso
├── routers/
│   ├── health.py            # GET /health, GET /ready
│   ├── targets.py           # GET /targets
│   ├── models.py            # GET /models, GET /models/{target}/{algorithm}, GET /models/{target}/{algorithm}/schema
│   └── predict.py           # POST /models/{target}/{algorithm}/predict, POST .../explain
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

### Preprocesamiento clínico en inferencia

El servicio `clinical_preprocessor.preprocess_raw()` transforma el objeto `patient` (variables clínicas crudas) al espacio de features del modelo antes de llamar a `predict_one()`. El flujo es:

1. **Descarte de vacíos y normalización estricta.** Se eliminan claves con valor `None`, `NaN`, cadena vacía o el literal `"nan"`. Algunos campos categóricos que el `OrdinalEncoder` del pipeline rechaza si reciben un valor desconocido (`Sexo`, `Atención`, `Cuello Móvil`, `Apertura Oral`, `Arritmia`) se normalizan a los valores exactos esperados (p. ej. `"masculino"` → `"M"`).
2. **Relleno de columnas ausentes del schema.** Todas las columnas declaradas en `raw_input_schema` que no vengan en el request se agregan con `None`, de modo que el cleaner no falle con `KeyError` en columnas opcionales.
3. **Limpieza determinista** (`clean_preop()` de `src/cleaning/`): mismas reglas que en entrenamiento — fechas, signos vitales, estandarización de texto de antecedentes, recálculo de estado nutricional desde el IMC, codificación multilabel de sistemas, etc.
4. **Enriquecimiento con IA** (`enrich_preop()`): codificación ICD de diagnósticos/procedimientos y estimación de severidad clínica mediante el modelo zero-shot **BART-MNLI** (HuggingFace). Los resultados se cachean en disco (`CACHE_DIR`) para no recomputar texto repetido.
5. **Reindexado e imputación** (`apply_imputation()`): se reordena al `manifest.feature_names` (las features no derivadas quedan `NaN`) y se imputan según `manifest.imputation` (`fill_constant` con centinela `-1`, o `fill_median` con la mediana de entrenamiento del manifest).

Esto significa que **todos los campos del paciente son opcionales en el request**: enviar más datos clínicos produce predicciones más confiables, pero el pipeline tolera ausencias e imputa lo que falte. La lista de campos aceptados y su descripción se obtiene mediante el endpoint `/schema`.

> **Nota de latencia:** la primera predicción que requiera enriquecimiento descarga los pesos de BART-MNLI (vía `HF_HOME`) y es lenta; las siguientes usan el modelo en memoria y la caché de texto en disco.

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

**Descripción:** Schema de **entrada clínica cruda** del modelo: la lista de campos que el cliente puede enviar (nombre de columna del dataset, tipo y descripción legible), más un ejemplo real de paciente. Permite a un frontend renderizar el formulario de entrada sin conocer las features internas del modelo. El schema proviene de `raw_input_schema` en el manifest (fuente única de verdad: `src/cleaning/schema.py`, 48 campos clínicos).

**Parámetros de ruta:** igual que `4.5`.

**Response (campos principales):**

| Campo | Tipo | Descripción |
|---|---|---|
| `model_id` | `string` | Identificador del modelo |
| `target` | `string` | Slug del target |
| `algorithm` | `string` | Algoritmo |
| `features` | `list[FeatureSpec]` | Lista ordenada de campos clínicos crudos (ver tabla abajo) |
| `threshold` | `float` | Umbral de decisión |
| `threshold_metric` | `string` | Métrica de optimización del threshold |
| `prevalence` | `object` | Prevalencia en train/test |
| `calibrated` | `bool` | Si está calibrado |
| `calibration_method` | `string\|null` | Método de calibración |
| `input_example` | `object\|null` | Ejemplo de paciente real (extraído del set de entrenamiento) listo para enviar a `/predict` |
| `warnings` | `list[string]` | Advertencias |

**Estructura de `FeatureSpec`:**

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | `string` | Nombre exacto de la columna clínica (tal como se envía en `patient`; p. ej. `"Examen_Hemoglobina(g/dl)"`) |
| `dtype` | `string` | Tipo de dato: `"float64"` (numérico) u `"object"` (texto/categórico) |
| `required` | `bool` | Siempre `false` — todos los campos son opcionales (el pipeline imputa lo ausente) |
| `description` | `string\|null` | Descripción legible y valores admitidos, para renderizar la UI |

**curl:**
```bash
curl http://localhost:8000/models/hospitalization_risk/xgboost/schema
```

---

### 4.7 `POST /models/{target}/{algorithm}/predict`

**Descripción:** Predicción individual. Recibe las **variables clínicas crudas** de un paciente, ejecuta internamente el pipeline completo de limpieza y enriquecimiento (ver §3) y retorna la clase predicha, la probabilidad calibrada y el nivel de riesgo.

**Parámetros de ruta:** igual que `4.5`.

**Request body:**

```json
{
  "patient": {
    "Edad": 35,
    "Sexo": "F",
    "Atención": "Electivo",
    "Peso (Kg)": 68,
    "Talla (cm)": 170,
    "IMC": 23.5,
    "Tensión Arterial Sistólica (mm/Hg)": 118,
    "Tensión Arterial Diastólica (mm/Hg)": 75,
    "Examen_Hemoglobina(g/dl)": 14.2,
    "Antecedentes cardiovasculares": "ninguno",
    "Antecedente endocrinológicos": "ninguno",
    "Dx Preoperatorio": "colelitiasis",
    "Procedimiento propuesto": "colecistectomía laparoscópica"
  }
}
```

El objeto `patient` acepta cualquier subconjunto de los campos clínicos crudos descritos en `/schema` (nombres de columna del dataset, en español). Los campos no incluidos se imputan automáticamente tras el pipeline. Las claves desconocidas se ignoran silenciosamente; no es necesario enviar las features internas del modelo.

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
    "patient": {
      "Edad": 35,
      "Sexo": "F",
      "Peso (Kg)": 68,
      "Talla (cm)": 170,
      "IMC": 23.5,
      "Examen_Hemoglobina(g/dl)": 14.2,
      "Dx Preoperatorio": "colelitiasis",
      "Procedimiento propuesto": "colecistectomía laparoscópica"
    }
  }'
```

**curl (paciente mínimo — imputa el resto):**
```bash
curl -X POST http://localhost:8000/models/hospitalization_risk/xgboost/predict \
  -H "Content-Type: application/json" \
  -d '{"patient": {"Edad": 78, "IMC": 33.8}}'
```

**Errores de negocio (HTTP 200, `success: false`):**

| `code` | Causa |
|---|---|
| `invalid_input` | Falta la clave `patient` o no es un objeto JSON |
| `preprocessing_error` | El pipeline de limpieza/enriquecimiento falló con los datos provistos |

**Errores HTTP:**

| Código | Causa |
|---|---|
| `404` | `target` o `algorithm` no registrado |

> **Nota:** no existe un endpoint de predicción en lote (`/predict/batch`). El rediseño hacia entrada clínica cruda con enriquecimiento por IA priorizó la predicción individual en el flujo de valoración preanestésica; el procesamiento masivo se realiza directamente con el pipeline de Airflow.

---

### 4.8 `POST /models/{target}/{algorithm}/explain`

**Descripción:** Explicabilidad por caso. Recibe las variables clínicas crudas de un paciente (igual que `/predict`) y retorna las **top-N contribuciones SHAP**: qué features empujaron la probabilidad hacia arriba o hacia abajo para ese paciente.

**Parámetros de ruta:** igual que `4.5`.

**Request body:**

```json
{
  "patient": {
    "Edad": 78,
    "IMC": 33.8,
    "Dx Preoperatorio": "estenosis aórtica severa",
    "Antecedentes cardiovasculares": "hipertensión, fibrilación auricular"
  },
  "top_n": 10
}
```

`top_n` es opcional (por defecto `10`) y se acota al rango `[1, n_features]`.

**Response:**

```json
{
  "success": true,
  "data": {
    "contributions": [
      { "feature": "score_proc_critical", "value": 1.0, "shap_value": 0.142 },
      { "feature": "Edad", "value": 78.0, "shap_value": 0.091 },
      { "feature": "Examen_Hemoglobina(g/dl)", "value": 9.1, "shap_value": 0.058 }
    ],
    "top_n": 10,
    "algorithm": "xgboost",
    "model_id": "target_f_predictibilidad_maxima__xgboost"
  },
  "errors": null,
  "meta": { "request_id": "...", "model_id": "target_f_predictibilidad_maxima__xgboost" }
}
```

| Campo (de cada contribución) | Tipo | Descripción |
|---|---|---|
| `feature` | `string` | Nombre de la feature interna del modelo |
| `value` | `float\|null` | Valor de esa feature para el paciente tras el preprocesamiento |
| `shap_value` | `float` | Contribución SHAP (positiva = empuja hacia clase 1; negativa = hacia clase 0) |

**curl:**
```bash
curl -X POST http://localhost:8000/models/hospitalization_risk/xgboost/explain \
  -H "Content-Type: application/json" \
  -d '{"patient": {"Edad": 78, "IMC": 33.8}, "top_n": 5}'
```

**Errores de negocio (HTTP 200, `success: false`):**

| `code` | Causa |
|---|---|
| `invalid_input` | Falta la clave `patient` o no es un objeto JSON |
| `shap_not_available` | La biblioteca SHAP no está instalada en el servidor |
| `explain_error` | Fallo al preprocesar o calcular los valores SHAP |

---

## 5. Configuración y variables de entorno

La aplicación se configura exclusivamente a través de variables de entorno. La plantilla completa se encuentra en `api/.env.example`.

| Variable | Por defecto | Descripción |
|---|---|---|
| `PROJECT_ROOT` | `.` (directorio de trabajo) | Raíz del proyecto. Se usa para resolver el resto de rutas si no se especifican. |
| `MODELS_DIR` | `$PROJECT_ROOT/output/v2/models` | Directorio raíz de modelos. La API escanea subdirectorios en busca de `*_manifest.json`. |
| `CACHE_DIR` | `$PROJECT_ROOT/cache` | Caché en disco del enriquecimiento clínico (ICD + severidad BART), compartida con el pipeline. |
| `CONFIG_DIR` | `$PROJECT_ROOT/config` | Directorio de configuración; la API carga `cleaning_config.yaml` para reproducir la limpieza de entrenamiento. |
| `HF_HOME` | (default de HuggingFace) | Directorio de caché de pesos de HuggingFace (BART-MNLI). En Docker se monta en el volumen `hf_cache`. |
| `LOG_LEVEL` | `INFO` | Nivel de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `CORS_ORIGINS` | `*` | Orígenes permitidos para CORS, separados por coma. Ejemplo: `http://localhost:3000,https://mi-dominio.com`. |
| `MODEL_CACHE_SIZE` | `8` | Número máximo de modelos cargados simultáneamente en memoria (LRU). |

> `MAX_BATCH_SIZE` permanece como ajuste reservado en `Settings`, pero no tiene efecto desde la eliminación del endpoint de predicción en lote.

Si `CONFIG_DIR` no contiene `cleaning_config.yaml`, la API arranca igualmente pero el preprocesamiento clínico queda deshabilitado (las predicciones fallarán con `preprocessing_error`).

Ejemplo de `.env.example`:
```
API_PORT=8000
API_LOG_LEVEL=INFO
API_CORS_ORIGINS=http://localhost:3000,https://midominio.com
API_MODEL_CACHE_SIZE=8
```

> Nota: en el `docker-compose.yaml` las variables se pasan sin el prefijo `API_` (p. ej. `LOG_LEVEL`, `CORS_ORIGINS`). El prefijo en `.env.example` es solo una convención para facilitar la carga con `--env-file`.

---

## 6. Despliegue con Docker

### Imagen (`api/Dockerfile`)

La imagen emplea un build multi-stage basado en `python:3.11-slim`. El stage `builder` instala las dependencias de `api/requirements.txt` (que ahora incluyen `transformers`, `torch`, `nltk`, `rapidfuzz`, `deep-translator` para el preprocesamiento clínico) y descarga el corpus de stopwords de NLTK. El stage `runtime` copia los paquetes instalados, los datos de NLTK (`NLTK_DATA=/home/api/nltk_data`) y **el código de `api/`, `src/` y `config/`** —necesario porque la API reutiliza el pipeline de limpieza/enriquecimiento de `src/cleaning/`—, ejecuta el proceso como usuario no-root (`api`, UID 1000), expone el puerto `8000` e incorpora un healthcheck interno con `curl -fs http://localhost:8000/health`.

Comando de inicio del contenedor:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Servicio `api` en `docker-compose.yaml`

El compose levanta el servicio `api` junto con Airflow y PostgreSQL. Monta `output/` (modelos), `config/` como volúmenes de solo lectura, `cache/` (caché de enriquecimiento, lectura-escritura) y un volumen nombrado `hf_cache` para los pesos de HuggingFace:

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
    CACHE_DIR: /app/cache
    CONFIG_DIR: /app/config
    HF_HOME: /app/hf_cache
    LOG_LEVEL: "${API_LOG_LEVEL:-INFO}"
    CORS_ORIGINS: "${API_CORS_ORIGINS:-*}"
    MODEL_CACHE_SIZE: "${API_MODEL_CACHE_SIZE:-8}"
  volumes:
    - ./output:/app/output:ro
    - ./cache:/app/cache
    - ./config:/app/config:ro
    - hf_cache:/app/hf_cache
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

> **Nota sobre el alcance:** con el rediseño hacia entrada clínica cruda (`4fa155e`), los antiguos tests E2E que enviaban features procesadas (predicción happy-path, validación de features, batch, latencia, robustez) quedaron obsoletos y fueron eliminados. La suite E2E vigente cubre salud y descubrimiento de modelos; el path completo de predicción cruda depende de la descarga de pesos de BART-MNLI y se valida manualmente o vía el frontend. La lógica de preprocesamiento y métricas se cubre en `tests/src/`.

### Archivos de test (API)

| Archivo | Tests | Qué cubren |
|---|---|---|
| `test_e2e_health.py` | 2 | `/health` y `/ready` |
| `test_e2e_discovery.py` | 4 | `/targets`, `/models`, `/models/{t}/{a}`, `/models/{t}/{a}/schema` |

### Cómo ejecutar los tests

```bash
# Suite completa del proyecto
pytest tests/ -v

# Solo E2E de la API (salud + descubrimiento)
pytest tests/api/ -v

# Tests unitarios del pipeline (incluye bootstrap CI, limpieza, etc.)
pytest tests/src/ -v

# Con variables de entorno explícitas
MODELS_DIR=output/v2/models pytest tests/api/ -v
```

Conteo actual aproximado: **6 tests E2E de API + 64 unitarios de pipeline + 21 de utilidades**.

> **Requisito:** los tests de descubrimiento asumen que `output/v2/models/` contiene al menos el modelo `hospitalization_risk/xgboost` (manifest + joblib). Si el modelo no está disponible, los tests que lo requieren se omiten automáticamente mediante `pytest.skip`.

### Fixture `conftest.py` — datos de ejemplo

El conftest define dos fixtures de **pacientes crudos** (claves = nombre de columna del dataset) listos para enviar a `/predict`:

**`low_risk_patient`** — paciente joven, sano, procedimiento menor electivo:
```json
{
  "Edad": 35, "Sexo": "Masculino", "Atención": "Electivo",
  "Peso (Kg)": 68, "Talla (cm)": 170, "IMC": 23.5,
  "Tipo de anestesia propuesta": "Local",
  "Examen_Hemoglobina(g/dl)": 14.2,
  "Dx Preoperatorio": "procedimiento menor electivo sin comorbilidades",
  "Procedimiento propuesto": "biopsia de piel"
}
```

**`high_risk_patient`** — paciente mayor, obeso, urgencia, comorbilidades graves:
```json
{
  "Edad": 78, "Sexo": "Masculino", "Atención": "Urgente",
  "Peso (Kg)": 92, "Talla (cm)": 165, "IMC": 33.8,
  "Tipo de anestesia propuesta": "General",
  "Examen_Hemoglobina(g/dl)": 9.1,
  "Dx Preoperatorio": "insuficiencia cardíaca congestiva descompensada, sepsis",
  "Procedimiento propuesto": "laparotomía exploratoria de emergencia",
  "Antecedentes cardiovasculares": "hipertensión arterial, fibrilación auricular",
  "Antecedente endocrinológicos": "diabetes mellitus tipo 2 insulinorrequiriente"
}
```
