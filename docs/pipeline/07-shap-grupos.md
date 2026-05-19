# Etapa 7 — Análisis SHAP por Grupos Clínicos (TP / FN / FP / TN)

**Código fuente:**
- [`src/evaluation/explainability.py`](../../src/evaluation/explainability.py) — `compute_shap_values`: calcula SHAP values usando TreeExplainer, LinearExplainer o KernelExplainer según el modelo
- [`src/evaluation/shap_groups.py`](../../src/evaluation/shap_groups.py) — `compute_shap_group_profiles`, `build_fn_insight_table`: perfil SHAP por grupo clínico
- [`src/reports/shap_plots.py`](../../src/reports/shap_plots.py) — `plot_shap_beeswarm`, `plot_shap_waterfall_fn`, `plot_shap_group_comparison`: generación de plots

**Outputs (pipeline v2):**
- `output/v2/reports/shap/{target}/shap_values_{modelo}.npy` — Matriz SHAP cruda (n_test × n_features)
- `output/v2/reports/shap/{target}/shap_expected_{modelo}.txt` — Valor base (probabilidad media del modelo sobre X_train)
- `output/v2/reports/shap/{target}/shap_features_{modelo}.txt` — Nombres de features en el orden del array SHAP
- `output/v2/reports/shap/{target}/shap_beeswarm_{modelo}.png` — Beeswarm plot global (top 20 features)
- `output/v2/reports/shap/{target}/fn_waterfall/waterfall_fn_{N:02d}_{modelo}.png` — Waterfall plot por caso falso negativo (hasta 10 casos, ordenados por probabilidad descendente)
- `output/v2/reports/shap/{target}/shap_group_profiles_{modelo}.csv` — SHAP medio por grupo clínico (top 20 features por |Delta TP-FN|)
- `output/v2/reports/shap/{target}/shap_fn_insight_{modelo}.csv` — Tabla de interpretación clínica de FN vs TP (top 10 features)
- `output/v2/reports/shap/{target}/shap_group_comparison_{modelo}.png` — Gráfico de barras horizontales TP vs FN con delta

> **Relación con el documento anterior:** El [documento 05-explicabilidad.md](05-explicabilidad.md) cubre la importancia global de features (nativa + permutación) y la tabla `explainability_cases_{modelo}.csv`. Este documento cubre el análisis **comparativo por grupo clínico** usando SHAP, que requiere los archivos `.npy` generados por la tarea `make_task_shap_plots`.

---

## 1. Propósito — ¿Por qué analizar SHAP por grupos clínicos?

En screening preanestésico, los errores tienen consecuencias asimétricas: un **falso negativo** — paciente de riesgo real que el modelo no detecta — puede conducir a una cirugía sin valoración adecuada, con potencial de complicaciones graves. Analizar el promedio de SHAP values separadamente para los grupos TP, FN, FP y TN permite responder preguntas que la importancia global no puede abordar: ¿qué señales distinguen a los verdaderos positivos de los falsos negativos? ¿Qué features caracterizan a los pacientes que el modelo identifica erróneamente con mayor frecuencia? Esta segmentación transforma el análisis de explicabilidad en una herramienta clínicamente accionable: identifica los perfiles de paciente donde el modelo tiene puntos ciegos sistemáticos, lo que orienta el diseño de reglas de rescate o la recolección de features adicionales para esos subgrupos.

---

## 2. Tipos de análisis implementados

### 2.1 Análisis global — Beeswarm y valores SHAP crudos

La tarea `make_task_shap_plots` del DAG ejecuta el análisis SHAP sobre el test set completo y genera:

**Cálculo de SHAP values (`compute_shap_values`):**

La función selecciona automáticamente el explainer según el tipo de modelo:

| Tipo de modelo | Explainer SHAP | Características |
|---|---|---|
| `RandomForestClassifier`, `ExtraTreesClassifier`, `XGBClassifier`, `LGBMClassifier`, `HistGradientBoostingClassifier` | `TreeExplainer` | Exacto y rápido. Usa `feature_perturbation="interventional"`, `model_output="probability"`. Para modelos calibrados (`CalibratedClassifierCV`) extrae el estimador base del primer fold. |
| `LogisticRegression` | `LinearExplainer` | Exacto. Calcula la contribución lineal de cada feature. |
| `MLPClassifier`, `StackingClassifier`, `VotingClassifier` | `KernelExplainer` | Aproximado. Muestrea hasta 200 filas del background (`X_train`). Más lento. |

El resultado es una matriz `shap_values` de forma `(n_test_samples, n_features)` donde cada valor representa la contribución de esa feature a la predicción de la clase positiva para ese paciente.

**Beeswarm plot (`plot_shap_beeswarm`):** muestra las top 20 features por importancia SHAP media absoluta. Cada punto representa un paciente del test set; el eje X indica la contribución SHAP (positiva: empuja hacia la clase positiva; negativa: reduce el riesgo estimado) y el color el valor de la feature en ese paciente (rojo = alto, azul = bajo).

**Persistencia de artefactos crudos (`save_shap_values`):** se almacenan tres archivos: `shap_values_{modelo}.npy` con la matriz completa como array numpy, `shap_expected_{modelo}.txt` con el valor base escalar (probabilidad media del modelo) y `shap_features_{modelo}.txt` con los nombres de features en el mismo orden que las columnas del array.

### 2.2 Waterfall plots para casos FN

`plot_shap_waterfall_fn` genera un waterfall plot para cada uno de los hasta 10 falsos negativos con mayor probabilidad predicha — los más cercanos al umbral de decisión, es decir, los casos que el modelo "casi detectó". Cada plot muestra el valor base `expected_value` (probabilidad media del modelo), la contribución acumulada de cada feature desde ese valor base hasta la probabilidad final predicha, con features en rojo empujando hacia la clase positiva y en azul reduciendo el riesgo, y el índice del caso en el título. Los FN se ordenan por `y_proba` descendente: el primero (`waterfall_fn_01_...`) es el FN de mayor probabilidad predicha y el décimo el de menor probabilidad — el caso más difícil para el modelo.

### 2.3 Análisis por grupo clínico — Perfiles SHAP

La tarea `make_task_shap_group_analysis` (que depende de `make_task_shap_plots`) ejecuta:

**`compute_shap_group_profiles` (`src/evaluation/shap_groups.py`):**
1. Lee los artefactos SHAP desde disco (`.npy`, `.txt`).
2. Lee `explainability_cases_{modelo}.csv` para obtener la columna `case_type` (TP/FN/FP/TN) de hasta 10 casos por grupo.
3. Mapea `case_index` (índice original del DataFrame) a posición posicional en el array SHAP usando el índice de `X_test.parquet`.
4. Calcula el SHAP medio por grupo para cada feature.
5. Calcula `Delta_TP_FN = SHAP_TP - SHAP_FN` para medir qué features distinguen más a los verdaderos positivos de los falsos negativos.
6. Ordena por `|Delta_TP_FN|` descendente y retorna las top 20 features.

**`build_fn_insight_table` (`src/evaluation/shap_groups.py`):**
Genera una tabla de lectura clínica para las top 10 features, con una columna `Interpretacion` en texto libre que explica si la feature protege a los FN (les baja la probabilidad), si la señal es débil en FN vs fuerte en TP, etc.

**`plot_shap_group_comparison` (`src/reports/shap_plots.py`):**
Genera un gráfico de dos paneles:
- **Panel izquierdo:** barras horizontales de SHAP medio para TP (verde) y FN (naranja), para las top 15 features por |Delta|. Las features donde TP y FN tienen SHAP de signo opuesto se destacan con sombreado púrpura.
- **Panel derecho:** barra del delta `SHAP_TP - SHAP_FN` con el valor numérico anotado. Verde = el TP tiene más señal positiva; naranja = el FN tiene más señal positiva que el TP.

---

## 3. Implementación — Tareas del DAG

Las tareas SHAP forman una cadena secuencial después de `explainability`:

```
explainability__{modelo}__{target}
        │
        ▼
shap_plots__{modelo}__{target}          (make_task_shap_plots)
        │   ├── compute_shap_values
        │   ├── save_shap_values        → .npy, .txt
        │   ├── plot_shap_beeswarm     → shap_beeswarm_{modelo}.png
        │   └── plot_shap_waterfall_fn → fn_waterfall/waterfall_fn_*.png
        │
        ▼
shap_group_analysis__{modelo}__{target} (make_task_shap_group_analysis)
        │   ├── load_shap_artifacts    ← lee .npy, .txt
        │   ├── compute_shap_group_profiles → shap_group_profiles_{modelo}.csv
        │   ├── build_fn_insight_table      → shap_fn_insight_{modelo}.csv
        │   └── plot_shap_group_comparison  → shap_group_comparison_{modelo}.png
        │
        ▼
model_plots__{modelo}__{target}
```

La tarea `make_task_shap_group_analysis` incluye manejo explícito de `FileNotFoundError`: si los artefactos `.npy` no existen porque la tarea `shap_plots` falló o fue omitida, la tarea emite una advertencia y finaliza sin error, permitiendo que el resto del pipeline continúe sin interrupción.

---

## 4. Estructura de los archivos CSV de salida

### 4.1 `shap_group_profiles_{modelo}.csv`

Generado por `compute_shap_group_profiles`. Contiene hasta 20 filas (top features por |Delta TP-FN|).

| Columna | Tipo | Descripción |
|---|---|---|
| `Feature` | str | Nombre de la feature |
| `SHAP_TP` | float | SHAP medio sobre los casos TP seleccionados (hasta 10). Positivo = empuja hacia clase positiva |
| `SHAP_FN` | float | SHAP medio sobre los casos FN seleccionados (hasta 10) |
| `SHAP_FP` | float | SHAP medio sobre los casos FP seleccionados (hasta 10) |
| `SHAP_TN` | float | SHAP medio sobre los casos TN seleccionados (hasta 10) |
| `Delta_TP_FN` | float | `SHAP_TP - SHAP_FN`. Positivo = TP tiene más señal en esta feature que FN |
| `AbsDelta` | float | `|Delta_TP_FN|` — valor de ordenación. Las features con mayor delta son las que más distinguen TP de FN |
| `N_TP` | int | Número de casos TP incluidos en el cálculo (≤10) |
| `N_FN` | int | Número de casos FN incluidos en el cálculo (≤10) |
| `N_FP` | int | Número de casos FP incluidos en el cálculo (≤10) |
| `N_TN` | int | Número de casos TN incluidos en el cálculo (≤10) |

### 4.2 `shap_fn_insight_{modelo}.csv`

Generado por `build_fn_insight_table`. Contiene las top 10 features del perfil, con interpretación en texto.

| Columna | Tipo | Descripción |
|---|---|---|
| `Feature` | str | Nombre de la feature |
| `SHAP_TP` | float | SHAP medio en verdaderos positivos |
| `SHAP_FN` | float | SHAP medio en falsos negativos |
| `Delta_TP_FN` | float | Diferencia TP − FN |
| `Interpretacion` | str | Texto automático explicando la dirección del efecto. Ej: "Protege al FN (baja riesgo −0.008), eleva al TP (+0.180)" |

---

## 5. Ejemplo real — `shap_group_profiles_random_forest.csv` (`target_d_v2_hosp`)

Las 10 features con mayor diferencia |Delta TP−FN| para el Random Forest en `target_d_v2_hosp` (de [`output/v2/reports/shap/target_d_v2_hosp/shap_group_profiles_random_forest.csv`](../../output/v2/reports/shap/target_d_v2_hosp/shap_group_profiles_random_forest.csv)):

| Feature | SHAP_TP | SHAP_FN | SHAP_FP | SHAP_TN | Delta_TP_FN |
|---|---|---|---|---|---|
| `Tipo de anestesia propuesta_raquidea` | +0.1801 | −0.0080 | −0.0114 | −0.0117 | **+0.1880** |
| `Dx Preoperatorio Code_O` | +0.0829 | 0.0000 | 0.0000 | 0.0000 | **+0.0829** |
| `Tipo de anestesia propuesta_general` | +0.0752 | −0.0008 | −0.0013 | −0.0009 | **+0.0760** |
| `score_proc_critical` | +0.0641 | −0.0044 | +0.0010 | −0.0021 | **+0.0685** |
| `score_proc_high_severity` | +0.0400 | −0.0025 | +0.0011 | −0.0047 | **+0.0424** |
| `Procedimiento propuesto Code_L` | +0.0398 | −0.0008 | −0.0011 | −0.0011 | **+0.0406** |
| `score_proc_medium_severity` | +0.0289 | −0.0027 | +0.0068 | −0.0034 | **+0.0316** |
| `Tipo de anestesia propuesta_sedacion` | +0.0180 | −0.0065 | −0.0018 | −0.0665 | **+0.0245** |
| `Edad` | +0.0177 | −0.0025 | −0.0031 | −0.0031 | **+0.0202** |
| `Dx Preoperatorio Code_S` | +0.0160 | −0.0027 | +0.0015 | +0.0028 | **+0.0187** |

Y el correspondiente `shap_fn_insight_random_forest.csv` (de [`output/v2/reports/shap/target_d_v2_hosp/shap_fn_insight_random_forest.csv`](../../output/v2/reports/shap/target_d_v2_hosp/shap_fn_insight_random_forest.csv)):

| Feature | SHAP_TP | SHAP_FN | Delta_TP_FN | Interpretacion |
|---|---|---|---|---|
| `Tipo de anestesia propuesta_raquidea` | +0.180 | −0.008 | +0.188 | Protege al FN (baja riesgo −0.008), eleva al TP (+0.180) |
| `Dx Preoperatorio Code_O` | +0.083 | 0.000 | +0.083 | Señal débil en FN (+0.000) vs fuerte en TP (+0.083) |
| `Tipo de anestesia propuesta_general` | +0.075 | −0.001 | +0.076 | Protege al FN (baja riesgo −0.001), eleva al TP (+0.075) |
| `score_proc_critical` | +0.064 | −0.004 | +0.068 | Protege al FN (baja riesgo −0.004), eleva al TP (+0.064) |

---

## 6. Interpretación clínica

### 6.1 Lectura del Delta TP−FN

Un `Delta_TP_FN` positivo y elevado en una feature indica que esa feature **empuja hacia la clase positiva con mucha más intensidad en los TP que en los FN**. Existen dos interpretaciones. La primera es que el FN simplemente no presenta esa condición: si `Tipo de anestesia propuesta_raquidea` tiene Delta=+0.188, los casos FN típicamente no van a anestesia raquídea — sus procedimientos son de un tipo diferente, con un perfil de riesgo que el modelo no captura bien. La segunda es que el FN presenta la condición pero el modelo la subestima: el SHAP del FN en esa feature es positivo pero insuficiente para superar el umbral de decisión.

### 6.2 Features con SHAP de signo opuesto

Cuando `SHAP_TP > 0` y `SHAP_FN < 0` para la misma feature, esa feature actúa en **direcciones opuestas** para ambos grupos: reduce el riesgo estimado de los FN mientras eleva el de los TP. En el ejemplo de `target_d_v2_hosp`, `Tipo de anestesia propuesta_raquidea` muestra SHAP_TP=+0.18 y SHAP_FN=−0.008: los TP van a anestesia raquídea, que el modelo asocia a mayor complejidad y riesgo, mientras que los FN van a otro tipo de anestesia, donde este indicador reduce el riesgo. El comportamiento de `Tipo de anestesia propuesta_sedacion` (SHAP_TP=+0.018, SHAP_FN=−0.0065) muestra un patrón similar: la sedación reduce el riesgo de forma marcada para los TN (SHAP_TN=−0.067), pero entre los TP algunos procedimientos bajo sedación resultan de riesgo. Estas features con signo opuesto aparecen resaltadas con sombreado púrpura en el gráfico comparativo de grupos.

### 6.3 Uso de los waterfall plots de FN

El waterfall del FN #1 — el caso con mayor probabilidad predicha — es el más relevante para auditoría: representa al paciente que el modelo "casi detectó". Si la feature que impidió superar el umbral es clínicamente relevante (hemoglobina, score de diagnóstico, tipo de procedimiento), eso orienta qué información adicional podría mejorar la detección de ese subgrupo. Por ejemplo, si el waterfall muestra que `Dx Preoperatorio Code_O` tiene SHAP=0 porque el paciente no pertenece a esa categoría diagnóstica, pero el score de severidad tiene SHAP positivo que casi alcanza el umbral, el clínico puede concluir que para ese perfil la señal reside en la severidad del diagnóstico, no en el tipo de procedimiento.

### 6.4 Puntos ciegos del modelo: subgrupos con SHAP_FN ≈ 0

Cuando una feature presenta `SHAP_FN ≈ 0` con `SHAP_TP` elevado, el modelo no detecta esa señal en los FN: esos pacientes de riesgo real no poseen las características que el modelo utiliza para elevar la probabilidad predicha. Estos subgrupos son candidatos a la recolección de features adicionales que capturen mejor su perfil de riesgo, al diseño de reglas de rescate clínicas basadas en variables no incluidas en el modelo actual, o a la reconsideración de la definición del target para ese subgrupo específico.

El análisis SHAP por grupos hace explícito lo que la importancia global oculta: un modelo puede alcanzar AUC alto en promedio mientras falla de forma sistemática en ciertos perfiles de pacientes de alto riesgo, precisamente los que más se beneficiarían de una intervención temprana.
