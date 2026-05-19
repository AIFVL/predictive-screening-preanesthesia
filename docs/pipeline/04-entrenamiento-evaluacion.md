# Etapa 4 — Entrenamiento y Evaluación de Modelos

**Código fuente:**
- [`src/models/trainer.py`](../../src/models/trainer.py) — Entrenamiento, calibración y evaluación
- [`src/models/registry.py`](../../src/models/registry.py) — Registro de modelos disponibles
- [`src/models/hyperparameter_search.py`](../../src/models/hyperparameter_search.py) — Búsqueda de hiperparámetros
- [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py) — Cálculo de métricas
- [`src/evaluation/comparison.py`](../../src/evaluation/comparison.py) — Comparación entre modelos
- [`src/evaluation/subgroups.py`](../../src/evaluation/subgroups.py) — Evaluación por subgrupos
- [`config/models_config.yaml`](../../config/models_config.yaml) — Configuración de todos los modelos

**Outputs (pipeline v2):**
- `output/v2/models/{target}/{modelo}_model.joblib` — Modelo entrenado serializado (calibrado cuando aplica)
- `output/v2/models/{target}/{modelo}_metrics.json` — Métricas finales en test (con threshold óptimo aplicado)
- `output/v2/models/{target}/{modelo}_eval.json` — Evaluación completa: bloque `test` + bloque `cv` (10-fold cross-validation, según `n_folds: 10` en `config/pipeline_config.yaml`)
- `output/v2/models/{target}/{modelo}_manifest.json` — Contrato del modelo: feature_names, dtypes, threshold, calibration, prevalence — consumido por la API
- `output/v2/plots/{target}/{modelo}_roc_pr.png` — Curvas ROC y PR
- `output/v2/plots/{target}/{modelo}_confusion.png` — Matriz de confusión
- `output/v2/plots/{target}/{modelo}_threshold.png` — Análisis de umbral
- `output/v2/reports/comparison_table.json` — Tabla agregada con métricas de todos los (target, modelo)

**Configuración:**
- [`config/models_config.yaml`](../../config/models_config.yaml) — Lista de modelos, hiperparámetros base, espacios de búsqueda, flag `calibrate`.
- [`config/pipeline_config.yaml`](../../config/pipeline_config.yaml) — Configuración global del pipeline (random_state, fracciones de split, etc.).

---

## 1. Configuración del entrenamiento

### 1.1 División del dataset

El dataset (`merged.parquet`, 23,387 registros) se divide en:
- **Train:** 80% → 18,709 registros
- **Test:** 20% → 4,678 registros

Las cifras de positivos dependen del target: para `target_d_v2_hosp` hay 5,180 positivos en train (27.7%); para `target_f_predictibilidad_maxima` la prevalencia es ~19.43%. El número de filas (18,709 / 4,678) aplica a ambos targets — comparten el mismo split.

La división es **estratificada** por el target, lo que garantiza que ambos conjuntos mantengan la misma proporción de positivos del target correspondiente. Esto es esencial para que las métricas de evaluación sobre el conjunto de prueba sean representativas del comportamiento real del modelo.

Los splits están fijados con `random_state=42` y almacenados en `output/v2/data_processed/{target}/splits/` para garantizar la reproducibilidad. Todos los modelos se entrenan sobre exactamente el mismo split por target, lo que hace que la comparación entre algoritmos sea directamente válida.

### 1.2 Preprocesamiento de features para el modelo

Antes de entrenar, las features seleccionadas se procesan:
1. **Conversión a numérico:** `pd.to_numeric(..., errors="coerce")` convierte valores no numéricos a NaN.
2. **Imputación de faltantes:** Se imputan con `-1` (valor centinela). El valor -1 fue elegido porque todos los valores válidos son ≥ 0 en este dataset — el modelo puede aprender que -1 significa "dato ausente".

La estrategia exacta se serializa en cada `<modelo>_manifest.json` bajo el campo `imputation`:
```json
{
  "strategy": "fill_constant",
  "value": -1
}
```

La API consume ese campo para imputar los inputs en producción exactamente igual que en entrenamiento — ver [08-api-inferencia.md](08-api-inferencia.md).

### 1.3 Desbalance de clases

El dataset presenta un desbalance de aproximadamente 2.6:1 (72.3% negativos, 27.7% positivos). Aunque no es extremo, es suficiente para sesgar a los modelos hacia predecir siempre la clase mayoritaria sin ajuste. La estrategia adoptada es `class_weight="balanced"` en los modelos de sklearn que lo soportan, ajustando los pesos de las muestras durante el entrenamiento de forma que los errores sobre la clase minoritaria (positivos) se penalicen proporcionalmente más. El peso asignado a cada clase es inversamente proporcional a su frecuencia:

```
peso_positivo = n_total / (2 × n_positivos) = 23387 / (2 × 6475) ≈ 1.81
peso_negativo = n_total / (2 × n_negativos) = 23387 / (2 × 16912) ≈ 0.69
```

XGBoost usa un enfoque equivalente: `scale_pos_weight = n_negativos / n_positivos ≈ 2.61`.

### 1.4 Calibración de probabilidades

Los modelos de árbol (Random Forest, Extra Trees, XGBoost, HGB, LightGBM) producen scores que no constituyen probabilidades bien calibradas: tienden a concentrarse en los extremos o en rangos que no corresponden a frecuencias reales de positivos. Para que el umbral de decisión tenga un significado probabilístico consistente — esto es, que threshold=0.17 corresponda efectivamente a "probabilidad de evento adverso ≥ 17%" — se aplica **calibración con `CalibratedClassifierCV`** (método isotónico) sobre el conjunto de entrenamiento.

En `models_config.yaml`, cada modelo lleva un campo `calibrate: true|false`; en v2 todos los modelos están configurados con `calibrate: true`. La calidad resultante de la calibración se evalúa con las métricas ECE, MCE y Brier — ver [06-calibracion.md](06-calibracion.md).

---

## 2. Modelos entrenados

En la versión v2 se entrenan **9 modelos** para cada versión del target, cubriendo un amplio espectro de familias algorítmicas. La configuración exacta vive en [`config/models_config.yaml`](../../config/models_config.yaml).

### 2.1 Regresión Logística
```yaml
C: 1.0
class_weight: balanced
solver: saga
max_iter: 3000
```

El modelo lineal más simple del conjunto, asume que la relación entre features y target es lineal en el espacio log-odds. Cumple una doble función: actúa como **baseline** de referencia y como **meta-aprendiz** en el modelo de stacking. Sus coeficientes son directamente interpretables, su entrenamiento es rápido y presenta bajo riesgo de sobreajuste; en contrapartida, no captura interacciones ni relaciones no lineales entre features.

### 2.2 Random Forest
```yaml
n_estimators: 300
max_depth: null  # árboles sin restricción de profundidad
class_weight: balanced
calibrate: true
```

Ensemble de 300 árboles de decisión entrenados sobre subconjuntos aleatorios de datos y features, cuya predicción final es el promedio de probabilidades de todos los árboles. Captura interacciones complejas, es robusto frente a outliers y produce importancias de features de forma natural; su principal desventaja es el tamaño del modelo resultante y la mayor lentitud de entrenamiento respecto a modelos lineales. La configuración `max_depth: null` permite que los árboles crezcan sin restricción de profundidad; con `class_weight=balanced` y el valor implícito de `min_samples_leaf`, esto no produce sobreajuste severo en la práctica.

### 2.3 Extra Trees (Extremely Randomized Trees)
```yaml
n_estimators: 300
max_depth: null
class_weight: balanced
calibrate: true
```

Variante de Random Forest donde la selección de splits se realiza de forma completamente aleatoria, sin buscar el corte óptimo en cada nodo. Esto genera árboles individualmente más ruidosos pero típicamente produce un ensemble más robusto. Respecto a Random Forest, Extra Trees suele ser más rápido de entrenar aunque con mayor varianza; en la práctica, las métricas de ambos son muy similares sobre este dataset.

### 2.4 XGBoost
```yaml
n_estimators: 300
learning_rate: 0.05
max_depth: 6
subsample: 0.8
colsample_bytree: 0.8
eval_metric: logloss
```

Gradient Boosting con árboles. A diferencia de Random Forest (que entrena árboles en paralelo), XGBoost los entrena secuencialmente: cada árbol corrige los errores del anterior. Es el estándar de facto para datasets tabulares en competiciones de ML.

**Hiperparámetros clave:**
- `learning_rate: 0.05` — cada árbol contribuye con el 5% de su predicción, evitando sobreajuste
- `max_depth: 6` — árboles poco profundos, capturan interacciones de orden ≤6
- `subsample: 0.8`, `colsample_bytree: 0.8` — muestreo aleatorio de datos y features por árbol, como regularización

### 2.5 HistGradientBoosting
```yaml
max_iter: 300
learning_rate: 0.05
max_depth: null
```

Implementación de sklearn del Gradient Boosting basada en histogramas. Es más eficiente en memoria y tiempo que XGBoost para datasets medianos, maneja NaN nativamente sin necesidad de imputación previa.

**Ventaja sobre XGBoost:** No requiere imputación explícita de faltantes — los maneja internamente como una categoría especial.

### 2.6 LightGBM *(añadido en v2)*
```yaml
n_estimators: 300
learning_rate: 0.05
max_depth: -1            # sin límite — controlado por num_leaves
num_leaves: 31
class_weight: balanced
```

Gradient Boosting basado en histogramas, similar a `HistGradientBoosting` pero con la implementación de Microsoft. En la práctica es el modelo con mejor F2 sobre `target_d_v2_hosp` (F2=0.677, AUC=0.761) y compite codo a codo con XGBoost en `target_f_predictibilidad_maxima`.

**Vs. HistGradientBoosting:** LightGBM permite `num_leaves` como hiperparámetro principal en lugar de `max_depth`, lo que da control más fino sobre la capacidad del árbol. En este dataset las dos implementaciones convergen en métricas casi idénticas.

### 2.7 MLP (Red Neuronal)
```yaml
hidden_layer_sizes: [128, 64]
activation: relu
max_iter: 500
early_stopping: true
validation_fraction: 0.1
```

Red neuronal feedforward de dos capas ocultas (128 y 64 neuronas). El mecanismo de `early_stopping` detiene el entrenamiento cuando la pérdida sobre el 10% de validación deja de mejorar, previniendo el sobreajuste. Con 18,709 muestras y 80 features, las redes neuronales no tienen ventaja estructural sobre los métodos de árbol para datos tabulares; su inclusión en el conjunto responde a su uso como referencia comparativa.

### 2.8 Stacking (meta-aprendizaje)
```yaml
estimadores base: [random_forest, xgboost, hist_gradient_boosting]
meta-estimador: logistic_regression
cv: 5
passthrough: false
```

El stacking emplea tres modelos base (RF, XGBoost, HGB) para generar predicciones, y un meta-estimador (regresión logística) aprende cómo combinarlas para producir la predicción final. Cada modelo base comete errores en distintos subconjuntos de datos — RF puede fallar donde XGBoost no, y viceversa —, y el meta-estimador aprende a compensar esos errores sistemáticos. La validación cruzada interna de 5 folds garantiza que el meta-estimador nunca accede a las predicciones generadas sobre los mismos datos que se usaron para entrenar cada estimador base, evitando fuga de datos. Con `passthrough: false`, el meta-estimador solo recibe las predicciones de los modelos base, no las features originales.

> **Nota:** El ensemble completo de stacking también se calibra con `CalibratedClassifierCV` (método isotónico), de forma independiente a la calibración que ya tienen los modelos base individualmente.

### 2.9 Voting (ensemble por voto)
```yaml
estimadores: [random_forest, xgboost, hist_gradient_boosting]
voting: soft
```

Promedio simple de las probabilidades de RF, XGBoost y HGB. `voting: "soft"` usa las probabilidades directamente (no el voto de clases). Es más simple que stacking pero frecuentemente tiene rendimiento similar.

---

## 3. Selección de umbral (threshold)

Una vez entrenados, los modelos producen probabilidades para cada paciente. Para convertir esas probabilidades en una decisión binaria (¿necesita valoración sí/no?), se necesita un umbral.

### 3.1 Criterio de optimización

El umbral se selecciona mediante la estrategia `optimize_for: "recall_constraint"` (configurada en `config/pipeline_config.yaml`): se maximiza la **Precisión** sujeto a la restricción **Recall ≥ 0.85**. Es decir, el threshold elegido es el más alto posible que mantiene el Recall por encima de 0.85, maximizando así la Precisión sin sacrificar la detección de positivos.

### 3.2 Thresholds resultantes (v2)

Para `target_d_v2_hosp` (extraído de [`output/v2/reports/comparison_table.json`](../../output/v2/reports/comparison_table.json)):

| Modelo | Threshold | Interpretación |
|--------|-----------|----------------|
| `extra_trees` | 0.17 | Clasifica como positivo si probabilidad ≥ 17% |
| `hist_gradient_boosting` | 0.17 | — |
| `lightgbm` | 0.18 | — |
| `logistic_regression` | 0.38 | Umbral más alto — modelo más conservador |
| `mlp` | 0.19 | — |
| `random_forest` | 0.17 | — |
| `stacking` | 0.34 | Umbral moderado |
| `voting` | 0.19 | — |
| `xgboost` | 0.17 | — |

Para `target_f_predictibilidad_maxima`:

| Modelo | Threshold |
|--------|-----------|
| `extra_trees` | 0.14 |
| `hist_gradient_boosting` | 0.13 |
| `lightgbm` | 0.13 |
| `logistic_regression` | 0.38 |
| `mlp` | 0.13 |
| `random_forest` | 0.13 |
| `stacking` | 0.34 |
| `voting` | 0.14 |
| `xgboost` | 0.14 |

Los umbrales bajos son consecuencia directa de la calibración isotónica: los modelos de árbol calibrados asignan probabilidades que reflejan la prevalencia real del dataset (19–28%), por lo que raramente superan 0.5 y el umbral que satisface Recall ≥ 0.85 cae en rangos bajos. La regresión logística opera con un umbral mayor (0.38) porque con `class_weight=balanced` sus probabilidades se concentran más cerca de 0.5, y el stacking requiere 0.34 porque su meta-estimador es también una regresión logística. El umbral de `target_f_predictibilidad_maxima` es aún más bajo (0.13–0.14) que el de `target_d_v2_hosp` (0.17–0.18) porque su menor prevalencia (19% vs. 28%) lleva a los modelos calibrados a producir probabilidades más bajas en promedio.

---

## 4. Resultados por modelo

### 4.1 Target `target_d_v2_hosp` (versión histórica)

Valores en el **test set** (4,678 registros, nunca vistos durante el entrenamiento). Datos exactos de [`output/v2/reports/comparison_table.json`](../../output/v2/reports/comparison_table.json):

| Modelo | ROC AUC | Recall | Precision | F2 | Threshold |
|--------|---------|--------|-----------|----|-----------|
| **lightgbm** | **0.7614** | 0.8533 | 0.3702 | **0.6767** | 0.18 |
| stacking | 0.7608 | 0.8629 | 0.3624 | 0.6761 | 0.34 |
| voting | 0.7602 | 0.8562 | 0.3650 | 0.6746 | 0.19 |
| random_forest | 0.7593 | 0.8600 | 0.3537 | 0.6686 | 0.17 |
| xgboost | 0.7591 | 0.8678 | 0.3587 | 0.6759 | 0.17 |
| hist_gradient_boosting | 0.7561 | 0.8649 | 0.3594 | 0.6750 | 0.17 |
| extra_trees | 0.7463 | 0.8668 | 0.3531 | 0.6715 | 0.17 |
| mlp | 0.7063 | 0.8639 | 0.3382 | 0.6591 | 0.19 |
| logistic_regression | 0.7051 | 0.8600 | 0.3337 | 0.6538 | 0.38 |

El mejor modelo es **LightGBM** (AUC 0.761, F2 0.677), seguido de cerca por stacking, voting y XGBoost (todos en AUC ~0.76). Los seis modelos basados en árboles presentan AUC casi idéntico (0.746–0.761), convergiendo al mismo techo de señal disponible en los datos. El Recall de todos los modelos se sitúa entre 0.85 y 0.87, cumpliendo la restricción de diseño. La Precisión baja (~0.35) es el tradeoff deliberado de priorizar Recall: de cada tres pacientes que el modelo marca como "necesitan valoración", aproximadamente uno la requiere efectivamente.

### 4.2 Target `target_f_predictibilidad_maxima` *(recomendado y servido por la API)*

| Modelo | ROC AUC | Recall | Precision | F2 | Threshold |
|--------|---------|--------|-----------|----|-----------|
| **xgboost** | **0.8608** | 0.8528 | 0.3708 | **0.6769** | 0.14 |
| stacking | 0.8588 | 0.8569 | 0.3587 | 0.6706 | 0.34 |
| random_forest | 0.8577 | 0.8762 | 0.3451 | 0.6700 | 0.13 |
| voting | 0.8569 | 0.8611 | 0.3529 | 0.6685 | 0.14 |
| lightgbm | 0.8543 | 0.8666 | 0.3396 | 0.6613 | 0.13 |
| extra_trees | 0.8509 | 0.8707 | 0.3490 | 0.6703 | 0.14 |
| hist_gradient_boosting | 0.8481 | 0.8583 | 0.3360 | 0.6548 | 0.13 |
| logistic_regression | 0.7876 | 0.8514 | 0.2914 | 0.6151 | 0.38 |
| mlp | 0.7838 | 0.8680 | 0.2792 | 0.6105 | 0.13 |

El AUC promedio de los modelos de árbol pasa de ~0.76 a ~0.85 al cambiar al target F, efecto directo de redefinir el target hacia los flags más predecibles. XGBoost lidera con AUC 0.861, seguido por stacking, random forest y voting (todos AUC > 0.85). Los modelos lineales (regresión logística y MLP) quedan sustancialmente por debajo: la mayor señal disponible en el target F es aprovechada con mayor eficiencia por los métodos basados en árboles. El Brier Score de XGBoost es 0.097 en target F frente a 0.155 de RF en target D, lo que indica una mejora sustancial tanto en discriminación como en calibración.

### 4.3 Métricas completas — XGBoost / target_f_predictibilidad_maxima (modelo recomendado)

Datos de [`output/v2/models/target_f_predictibilidad_maxima/xgboost_metrics.json`](../../output/v2/models/target_f_predictibilidad_maxima/xgboost_metrics.json) y `xgboost_eval.json`:

```json
{
  "test": {
    "ROC_AUC": 0.8608,
    "PR_AUC": 0.7039,
    "Recall": 0.8528,
    "Precision": 0.3708,
    "F1": 0.5169,
    "F2": 0.6769,
    "Balanced_Accuracy": 0.7519,
    "Specificity": 0.6511,
    "Accuracy": 0.6903,
    "Brier": 0.0967,
    "FN_Rate": 0.1472,
    "Predicted_Positive_Rate": 0.4468,
    "Threshold": 0.14
  },
  "cv": {
    "ROC_AUC_mean": 0.8535, "ROC_AUC_std": 0.0135,
    "F2_mean": 0.6741, "F2_std": 0.0193,
    "Recall_mean": 0.863, "Recall_std": 0.0097,
    "Precision_mean": 0.3604, "Precision_std": 0.0252
  }
}
```

**Interpretación de las métricas principales:**

- **ROC AUC = 0.861 (test) / 0.854 (cv):** Dado un paciente positivo y uno negativo elegidos al azar, el modelo asigna mayor probabilidad al positivo el 86.1% de las veces. La consistencia entre test y validación cruzada (diferencia ± 0.014) indica que el modelo no está sobreajustado al split.

- **PR AUC = 0.704:** El área bajo la curva Precisión-Recall. Con una prevalencia de 0.194, el baseline aleatorio correspondería a PR AUC ≈ 0.194; el valor observado (0.704) representa una mejora sustancial.

- **Recall = 0.853:** El modelo detecta el 85.3% de los pacientes que necesitarían valoración formal, con una tasa de falsos negativos de 14.7%.

- **Precision = 0.371:** De los pacientes marcados como "requieren valoración", el 37.1% realmente lo necesita. La precisión es la métrica más afectada por el umbral bajo y refleja el tradeoff deliberado de diseño.

- **F2 = 0.677:** Métrica de optimización del umbral. Mejora marginalmente sobre `target_d_v2_hosp` pese al salto en AUC, porque F2 está acotado por la prevalencia y la restricción de mantener Recall ≥ 0.85.

- **Specificity = 0.651:** El modelo identifica correctamente el 65.1% de los pacientes que no requieren valoración, una mejora de 25 puntos sobre el target D (0.40). Esto se traduce directamente en una reducción de la carga de valoraciones innecesarias.

- **Predicted Positive Rate = 0.447:** El modelo remite a valoración al 44.7% de los pacientes, frente al 67% del target D, manteniendo un Recall equivalente. La mayor señal del target F permite discriminar mejor sin sacrificar sensibilidad.

- **Brier Score = 0.097:** Un modelo aleatorio con prevalencia 0.194 tendría Brier ≈ 0.156; el valor observado confirma que el modelo combina buena discriminación con probabilidades calibradas, interpretables como estimaciones de riesgo real. Los detalles de calibración se desarrollan en [06-calibracion.md](06-calibracion.md).

---

## 5. Análisis de las curvas

### 5.1 Curva ROC

La curva ROC (Receiver Operating Characteristic) muestra el tradeoff entre Recall (TPR) y FP rate (1-Specificity) para todos los posibles thresholds. AUC = 0.759 significa que la curva está un 75.9% del camino entre la diagonal (modelo aleatorio, AUC=0.5) y la esquina superior izquierda (modelo perfecto, AUC=1.0).

Los gráficos de curvas ROC están en `output/v2/plots/{target}/{modelo}_roc_pr.png`.

### 5.2 Curva Precision-Recall

La curva PR muestra el tradeoff entre Precisión y Recall para todos los thresholds. Es más informativa que ROC para datasets desbalanceados porque el baseline (línea horizontal en prevalencia=0.277) es visible y hace obvio cuánta mejora aporta el modelo sobre el azar.

### 5.3 Análisis de threshold

Los gráficos `{modelo}_threshold.png` muestran cómo varían Recall, Precisión, F1 y F2 al cambiar el threshold. El threshold de 0.17 está en la zona donde:
- Recall ≈ 0.86 (primer umbral que satisface Recall ≥ 0.85; sube el threshold y el Recall cae)
- Precisión es 0.35 (máxima Precisión con la restricción de Recall ≥ 0.85)

---

## 6. Matrices de confusión (test set, 4,678 pacientes)

### `target_d_v2_hosp` — Random Forest (threshold=0.17)

```
                    Predicho Neg.    Predicho Pos.
Real Negativo (N=3383):    1348           2035     → FP rate = 60.2%
Real Positivo (N=1295):     181           1114     → FN rate = 14.0%
```

### `target_f_predictibilidad_maxima` — XGBoost (threshold=0.14)

A partir de `Recall=0.853`, `Specificity=0.651`, prevalencia 0.194:

```
                    Predicho Neg.    Predicho Pos.
Real Negativo (N≈3769):    2454           1315     → FP rate = 34.9%
Real Positivo (N≈909):      134            775     → FN rate = 14.7%
```

El target F clasifica positivos en un 44.7% de los pacientes (vs. 67% en el target D) manteniendo el Recall en ~0.85. Es decir: identifica casi la misma fracción de positivos, pero con muchos menos falsos positivos, gracias a la mayor señal disponible.

---

## 7. ¿Por qué los modelos convergen dentro de cada target?

Dentro de un mismo target, RF, XGBoost, HGB, LightGBM, stacking y voting difieren en menos de 0.015 en AUC. Esta convergencia refleja la **señal limitada disponible en los datos preoperatorios**: con una MI máxima de 0.10 y una correlación de Pearson máxima de 0.23 para `target_d_v2_hosp`, todos los modelos extraen esencialmente la misma información. La diferencia entre un Random Forest y un Gradient Boosting resulta marginal cuando la señal disponible es baja, con independencia de la sofisticación algorítmica.

Al cambiar el target, en cambio, todos los modelos mejoran simultáneamente: el AUC promedio de los modelos de árbol sube de ~0.757 con `target_d_v2_hosp` a ~0.853 con `target_f_predictibilidad_maxima`. Este salto de ~0.10 puntos en AUC es de un orden de magnitud mayor que cualquier diferencia entre algoritmos y confirma el principio central del proyecto: el target define el techo de rendimiento alcanzable; el algoritmo solo determina qué tan cerca se llega a ese techo.

Este diagnóstico está documentado en detalle en el [análisis posoperatorio](../analisis-posoperatorio/README.md) y en el documento de [selección de features](03-seleccion-features.md).
