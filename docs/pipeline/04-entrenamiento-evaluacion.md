# Etapa 4 — Entrenamiento y Evaluación de Modelos

**Código fuente:**
- [`src/models/trainer.py`](../../src/models/trainer.py) — Entrenamiento, calibración y evaluación
- [`src/models/registry.py`](../../src/models/registry.py) — Registro de modelos disponibles
- [`src/models/hyperparameter_search.py`](../../src/models/hyperparameter_search.py) — Búsqueda de hiperparámetros
- [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py) — Cálculo de métricas
- [`src/evaluation/comparison.py`](../../src/evaluation/comparison.py) — Comparación entre modelos
- [`src/evaluation/subgroups.py`](../../src/evaluation/subgroups.py) — Evaluación por subgrupos
- [`config/models_config.yaml`](../../config/models_config.yaml) — Configuración de todos los modelos

**Outputs:**
- `output/v1/models/{target}/{modelo}_model.joblib` — Modelo entrenado serializado
- `output/v1/models/{target}/{modelo}_metrics.json` — Métricas en test
- `output/v1/plots/{target}/{modelo}_roc_pr.png` — Curvas ROC y PR
- `output/v1/plots/{target}/{modelo}_confusion.png` — Matriz de confusión
- `output/v1/plots/{target}/{modelo}_threshold.png` — Análisis de umbral

---

## 1. Configuración del entrenamiento

### 1.1 División del dataset

El dataset (`merged.parquet`, 23,387 registros) se divide en:
- **Train:** 80% → 18,709 registros (5,180 positivos, 27.7%)
- **Test:** 20% → 4,678 registros (1,295 positivos, 27.7%)

La división es **estratificada** por el target, garantizando que ambos conjuntos tengan la misma proporción de positivos (27.7%). Esto es crítico para que las métricas del test sean representativas.

Los splits están fijos (`random_state=42`) y guardados en `output/v1/data_processed/target_d_v2_hosp/splits/` para reproducibilidad. Todos los modelos se entrenan sobre exactamente el mismo split, haciendo la comparación justa.

### 1.2 Preprocesamiento de features para el modelo

Antes de entrenar, las 80 features seleccionadas se procesan:
1. **Conversión a numérico:** `pd.to_numeric(..., errors="coerce")` convierte valores no numéricos a NaN.
2. **Imputación de faltantes:** Se imputan con `-1` (valor centinela). El valor -1 fue elegido porque todos los valores válidos son ≥ 0 en este dataset — el modelo puede aprender que -1 significa "dato ausente".

### 1.3 Desbalance de clases

El dataset tiene 72.3% negativos y 27.7% positivos — un desbalance de aproximadamente 2.6:1. No es extremo, pero es suficiente para sesgar a los modelos hacia predecir siempre negativo.

**Estrategia adoptada:** `class_weight="balanced"` en los modelos de sklearn que lo soportan. Esto ajusta los pesos de las muestras durante el entrenamiento de modo que el modelo penaliza más los errores en la clase minoritaria (positivos). El peso de cada clase es inversamente proporcional a su frecuencia:

```
peso_positivo = n_total / (2 × n_positivos) = 23387 / (2 × 6475) ≈ 1.81
peso_negativo = n_total / (2 × n_negativos) = 23387 / (2 × 16912) ≈ 0.69
```

XGBoost usa un enfoque equivalente: `scale_pos_weight = n_negativos / n_positivos ≈ 2.61`.

### 1.4 Calibración de probabilidades

Los modelos de árbol (Random Forest, Extra Trees, XGBoost, HGB) producen scores que no son probabilidades bien calibradas — tienden a concentrarse en los extremos (0 o 1) o en rangos específicos que no corresponden a probabilidades reales.

Para que el threshold tenga un significado probabilístico consistente (threshold=0.17 significa "probabilidad de complicación ≥ 17%"), se aplica **calibración con `CalibratedClassifierCV`** (método isotónico) sobre el training set.

Los modelos logística, stacking y voting no necesitan calibración adicional porque ya producen probabilidades razonablemente calibradas.

---

## 2. Modelos entrenados

Se entrenaron **8 modelos** para cada versión del target, cubriendo un amplio espectro de familias algorítmicas:

### 2.1 Regresión Logística
```yaml
C: 1.0
class_weight: balanced
solver: lbfgs
max_iter: 1000
```

El modelo lineal más simple. Asume que la relación entre features y target es lineal en el espacio log-odds. Sirve como **baseline** y como **meta-aprendiz** en el stacking.

**Ventajas:** Coeficientes directamente interpretables, entrenamiento rápido, bajo riesgo de sobreajuste.
**Desventajas:** No captura interacciones ni relaciones no lineales entre features.

### 2.2 Random Forest
```yaml
n_estimators: 300
max_depth: null  # árboles sin restricción de profundidad
class_weight: balanced
calibrate: true
```

Ensemble de 300 árboles de decisión entrenados sobre subconjuntos aleatorios de datos y features. El resultado final es el promedio de las probabilidades de cada árbol.

**Ventajas:** Captura interacciones complejas, robusto a outliers, produce importancias de features naturalmente, bajo riesgo de sobreajuste por el promediado.
**Desventajas:** Modelos grandes (300 árboles × profundidad ilimitada), lento de entrenar, menos interpretable que regresión logística.

**Nota:** `max_depth: null` permite que los árboles crezcan sin restricción hasta haber separado correctamente todos los nodos de entrenamiento — con `class_weight=balanced` y `min_samples_leaf` implícito, esto no causa sobreajuste severo en la práctica.

### 2.3 Extra Trees (Extremely Randomized Trees)
```yaml
n_estimators: 300
max_depth: null
class_weight: balanced
calibrate: true
```

Variante de Random Forest donde la selección de splits también es aleatoria (no se busca el split óptimo). Esto hace los árboles más ruidosos individualmente pero el ensemble generalmente más robusto.

**Vs. Random Forest:** Extra Trees tiende a ser más rápido de entrenar pero con mayor varianza. En la práctica, las métricas son muy similares a RF.

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

Implementación de sklearn del Gradient Boosting basada en histogramas (similar a LightGBM). Es más eficiente en memoria y tiempo que XGBoost para datasets medianos, maneja NaN nativamente sin necesidad de imputación previa.

**Ventaja sobre XGBoost:** No requiere imputación explícita de faltantes — los maneja internamente como una categoría especial.

### 2.6 MLP (Red Neuronal)
```yaml
hidden_layer_sizes: [128, 64]
activation: relu
max_iter: 500
early_stopping: true
validation_fraction: 0.1
```

Red neuronal feedforward de dos capas ocultas (128 y 64 neuronas). El `early_stopping` detiene el entrenamiento cuando la pérdida en el 10% de validación no mejora, evitando sobreajuste.

**Limitaciones en este contexto:** Con 18,709 muestras y 80 features, una red neuronal no tiene ventaja sobre los métodos de árbol. Las redes neuronales brillan con datos de alta dimensionalidad (imágenes, texto) o volúmenes enormes. Aquí sirve de referencia.

### 2.7 Stacking (meta-aprendizaje)
```yaml
estimadores base: [random_forest, xgboost, hist_gradient_boosting]
meta-estimador: logistic_regression
cv: 5
passthrough: false
```

El stacking usa 3 modelos base (RF, XGBoost, HGB) para generar predicciones, y luego un meta-estimador (regresión logística) aprende cómo combinar esas predicciones para producir la predicción final.

**¿Por qué funciona?** Cada modelo base comete errores en distintos ejemplos — RF puede fallar donde XGBoost no, y viceversa. El meta-estimador aprende a "corregir" los errores sistemáticos de los modelos base. El CV interno (5-fold) asegura que el meta-estimador no ve las predicciones del training set en el mismo fold que se usó para entrenarlos (previene fuga de datos).

`passthrough: false` significa que el meta-estimador solo ve las predicciones de los modelos base, no las features originales.

### 2.8 Voting (ensemble por voto)
```yaml
estimadores: [random_forest, xgboost, hist_gradient_boosting]
voting: soft
```

Promedio simple de las probabilidades de RF, XGBoost y HGB. `voting: "soft"` usa las probabilidades directamente (no el voto de clases). Es más simple que stacking pero frecuentemente tiene rendimiento similar.

---

## 3. Selección de umbral (threshold)

Una vez entrenados, los modelos producen probabilidades para cada paciente. Para convertir esas probabilidades en una decisión binaria (¿necesita valoración sí/no?), se necesita un umbral.

### 3.1 Criterio de optimización

El umbral se optimiza para maximizar el **F2-score** en el training set (con validación cruzada implícita). F2 da el doble de peso al Recall que a la Precisión:

```
F2 = (1 + 2²) × (Precision × Recall) / (2² × Precision + Recall)
   = 5 × (Precision × Recall) / (4 × Precision + Recall)
```

Un F2 alto requiere Recall alto — el modelo debe detectar la mayoría de los positivos, aunque eso implique muchos falsos positivos.

### 3.2 Thresholds resultantes

| Modelo | Threshold | Interpretación |
|--------|-----------|----------------|
| `extra_trees` | 0.17 | Clasifica como positivo si probabilidad ≥ 17% |
| `hist_gradient_boosting` | 0.17 | — |
| `logistic_regression` | 0.38 | Umbral más alto — modelo más conservador |
| `mlp` | 0.19 | — |
| `random_forest` | 0.17 | — |
| `stacking` | 0.34 | Umbral moderado |
| `voting` | 0.19 | — |
| `xgboost` | 0.17 | — |

**¿Por qué 0.17?** Los modelos de árbol (calibrados) asignan probabilidades más bajas en general que los lineales. Un threshold de 0.17 no significa que el modelo sea "poco seguro" — significa que los modelos calibrados asignan probabilidades más distribuidas y el threshold óptimo para F2 cae en ese rango.

La regresión logística necesita 0.38 porque sus probabilidades están distribuidas de manera diferente — con `class_weight=balanced`, el modelo logístico tiende a asignar probabilidades más cercanas a 0.5 para los casos ambiguos.

---

## 4. Resultados por modelo

### 4.1 Target seleccionado: `target_d_v2_hosp`

Esta es la versión de mejor rendimiento. Los valores corresponden a evaluación en el **test set** (4,678 registros, nunca vistos durante el entrenamiento).

| Modelo | ROC AUC | Recall | Precision | F1 | F2 | Threshold |
|--------|---------|--------|-----------|----|----|-----------|
| stacking | **0.7608** | 0.8629 | 0.348 | 0.496 | 0.674 | 0.34 |
| voting | **0.7602** | 0.8562 | 0.352 | 0.498 | 0.671 | 0.19 |
| random_forest | **0.7593** | 0.8600 | 0.354 | 0.501 | 0.669 | 0.17 |
| xgboost | 0.7591 | 0.8678 | 0.347 | 0.494 | 0.676 | 0.17 |
| hist_gradient_boosting | 0.7561 | 0.8649 | 0.352 | 0.499 | 0.675 | 0.17 |
| extra_trees | 0.7463 | 0.8668 | 0.344 | 0.491 | 0.672 | 0.17 |
| mlp | 0.7063 | 0.8639 | 0.342 | 0.488 | 0.659 | 0.19 |
| logistic_regression | 0.7051 | 0.8600 | 0.317 | 0.462 | 0.654 | 0.38 |

**Observaciones:**
- La diferencia entre el mejor (stacking, 0.7608) y el peor (logística, 0.7051) es de apenas 0.055 en AUC. No hay una diferencia dramática entre algoritmos.
- Los 5 modelos basados en árboles (RF, XGBoost, HGB, ET, stacking, voting) tienen AUC casi idéntico (0.746–0.761). Todos convergen al mismo "techo de señal" disponible en los datos.
- El Recall de todos los modelos está en 0.85–0.87, que era la meta principal del diseño.
- La Precisión es baja (~0.35): de cada 3 pacientes que el modelo dice "sí necesitan valoración", solo 1 realmente la necesita. Esto es un tradeoff aceptable dada la prioridad de no perder verdaderos positivos.

### 4.2 Comparación entre versiones del target

Los modelos de árbol convergieron al mismo rendimiento independientemente del algoritmo. La diferencia real es entre versiones del target:

| Target | AUC promedio modelos árbol | Prevalencia | N positivos |
|--------|---------------------------|-------------|-------------|
| `target_d_v2` | ~0.636 | 16.93% | 3,961 |
| `target_d_v2_hosp` | ~0.757 | 27.69% | 6,475 |
| `target_d_v5` | ~0.759 | 25.63% | 5,997 |

El salto de 0.636 a 0.757 al pasar de `target_d_v2` a `target_d_v2_hosp` confirma que añadir `flag_hospitalizacion_no_anticipada` al target fue la decisión más impactante del proyecto — más que cualquier elección algorítmica.

### 4.3 Métricas completas del modelo seleccionado (Random Forest, target_d_v2_hosp)

```json
{
  "ROC_AUC": 0.7593,
  "PR_AUC": 0.6229,
  "Recall": 0.8600,
  "Precision": 0.3537,
  "F1": 0.5013,
  "F2": 0.6686,
  "Balanced_Accuracy": 0.6292,
  "Specificity": 0.3984,
  "Accuracy": 0.5262,
  "Brier": 0.1559,
  "FN_Rate": 0.1400,
  "Predicted_Positive_Rate": 0.6731,
  "Threshold": 0.17
}
```

**Interpretación de cada métrica:**

- **ROC AUC = 0.759:** Si tomamos un paciente positivo y un paciente negativo al azar, el modelo asigna mayor probabilidad al positivo el 75.9% de las veces. Discriminación moderada-buena.

- **PR AUC = 0.623:** El área bajo la curva Precision-Recall. Esta métrica es más informativa que ROC AUC cuando el dataset está desbalanceado — el baseline aleatorio tendría PR AUC ≈ 0.277 (prevalencia del target). PR AUC 0.623 representa una mejora sustancial sobre el azar.

- **Recall = 0.860:** El modelo detecta el 86% de los pacientes que sí necesitarían valoración formal. Solo el 14% de los verdaderos positivos son "perdidos" por el modelo (FN rate = 0.14).

- **Precision = 0.354:** De los pacientes que el modelo marca como "necesitan valoración", solo el 35.4% realmente la necesitan. El resto (64.6%) son falsos positivos — pacientes enviados a valoración innecesariamente.

- **F2 = 0.669:** Métrica objetivo, da doble peso al Recall. Valor razonable dado el desbalance.

- **Specificity = 0.398:** El modelo solo identifica correctamente el 39.8% de los pacientes que NO necesitarían valoración. El 60.2% de los negativos son clasificados como positivos (FP rate). Esto es el precio de mantener Recall alto.

- **Accuracy = 0.526:** Métrica engañosa aquí — el 52.6% de aciertos no es mejor que un modelo que siempre predice "positivo" (que tendría 72.3% de accuracy). La Accuracy no es útil cuando el dataset está desbalanceado.

- **Brier Score = 0.156:** Error cuadrático medio de las probabilidades. Un modelo perfecto tendría Brier=0; un modelo aleatorio tendría Brier ≈ 0.277 × (1-0.277) ≈ 0.2. Brier 0.156 indica probabilidades razonablemente bien calibradas.

- **Predicted Positive Rate = 0.673:** El modelo clasifica el 67.3% de todos los pacientes como positivos (con threshold=0.17). Esto refleja el threshold muy bajo — es deliberado para mantener Recall alto.

---

## 5. Análisis de las curvas

### 5.1 Curva ROC

La curva ROC (Receiver Operating Characteristic) muestra el tradeoff entre Recall (TPR) y FP rate (1-Specificity) para todos los posibles thresholds. AUC = 0.759 significa que la curva está un 75.9% del camino entre la diagonal (modelo aleatorio, AUC=0.5) y la esquina superior izquierda (modelo perfecto, AUC=1.0).

Los gráficos de curvas ROC están en `output/v1/plots/target_d_v2_hosp/{modelo}_roc_pr.png`.

### 5.2 Curva Precision-Recall

La curva PR muestra el tradeoff entre Precisión y Recall para todos los thresholds. Es más informativa que ROC para datasets desbalanceados porque el baseline (línea horizontal en prevalencia=0.277) es visible y hace obvio cuánta mejora aporta el modelo sobre el azar.

### 5.3 Análisis de threshold

Los gráficos `{modelo}_threshold.png` muestran cómo varían Recall, Precisión, F1 y F2 al cambiar el threshold. El threshold de 0.17 está en la zona donde:
- Recall ≈ 0.86 (se empieza a caer precipitadamente si se sube el threshold)
- F2 está cerca de su máximo
- Precisión es 0.35 (baja pero aceptable para un sistema de screening)

---

## 6. Matriz de confusión (threshold=0.17, test set)

Para el Random Forest en `target_d_v2_hosp`:

```
                    Predicho Neg.    Predicho Pos.
Real Negativo (N=3383):    1348           2035     → FP rate = 60.2%
Real Positivo (N=1295):     181           1114     → FN rate = 14.0%
```

En términos prácticos sobre el test set de 4,678 pacientes:
- **1,114 verdaderos positivos:** Pacientes que el modelo correctamente identifica como de riesgo.
- **181 falsos negativos:** Pacientes de riesgo que el modelo no detecta — los más preocupantes.
- **2,035 falsos positivos:** Pacientes enviados innecesariamente a valoración — carga adicional al sistema.
- **1,348 verdaderos negativos:** Pacientes que el modelo correctamente identifica como de bajo riesgo.

---

## 7. ¿Por qué todos los modelos convergen?

Una pregunta natural es: ¿por qué no hay un modelo claramente mejor? Random Forest, XGBoost, HGB y stacking difieren en menos de 0.015 en AUC. La respuesta es la **señal limitada en los datos**:

Con las features disponibles (preoperatorias), la MI máxima con el target es 0.10 y la correlación máxima de Pearson es 0.23. Todos los modelos están extrayendo la misma señal limitada. La diferencia entre un Random Forest y un Gradient Boosting es marginal cuando la señal disponible es baja — ambos aprenden el mismo patrón de correlaciones débiles.

El verdadero cuello de botella no es el algoritmo sino:
1. La definición del target (mezcla complicaciones predecibles con impredecibles)
2. Las features disponibles (no capturan todos los factores de riesgo relevantes)

Este diagnóstico está detallado en el [análisis posoperatorio](../analisis-posoperatorio/README.md) y en el documento de [selección de features](03-seleccion-features.md).
