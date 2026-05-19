# Etapa 6 — Calibración de Probabilidades

**Código fuente:**
- [`src/evaluation/calibration.py`](../../src/evaluation/calibration.py) — Cálculo de la curva de calibración y métricas (ECE, MCE, Brier)
- [`src/reports/calibration_plots.py`](../../src/reports/calibration_plots.py) — Reliability diagrams individuales y comparativos
- [`src/models/registry.py`](../../src/models/registry.py) — Tabla `ALGORITHM_CALIBRATION_METHODS` y función `fit_with_calibration`
- [`src/models/trainer.py`](../../src/models/trainer.py) — Integración de la calibración en el flujo de entrenamiento

**Outputs (pipeline v2):**
- `output/v2/reports/calibration/{target}/calibration_{modelo}.json` — Métricas y curva de calibración por modelo
- `output/v2/reports/calibration/{target}/calibration_{modelo}.png` — Reliability diagram individual (PNG)
- `output/v2/reports/calibration/{target}/calibration_comparison.png` — Comparación de todos los modelos en un único reliability diagram

> **Nota:** La calibración ocurre en **dos momentos distintos** del pipeline. Durante el entrenamiento (`fit_with_calibration` en `trainer.py`), los modelos se envuelven en `CalibratedClassifierCV` para corregir sus probabilidades crudas. Después de la evaluación, la tarea `make_task_calibration` del DAG mide la calidad de esa calibración sobre el test set y genera los reportes de esta etapa.

---

## 1. ¿Qué es la calibración y por qué importa en screening clínico?

Un modelo de ML puede discriminar bien — AUC alto — y al mismo tiempo producir probabilidades mal calibradas: por ejemplo, asignar p=0.8 a casos que en la realidad son positivos solo el 30% de las veces. En screening preanestésico esto tiene consecuencias directas, porque la probabilidad predicha es el fundamento con el que el equipo clínico justifica la remisión de un paciente a valoración adicional. Una probabilidad calibrada de 0.25 puede interpretarse como "25% de riesgo de evento adverso" y usarse para priorizar la atención; una probabilidad no calibrada carece de ese significado operativo. Los modelos de árbol (Random Forest, XGBoost, Extra Trees) son especialmente propensos a probabilidades concentradas en los extremos o sesgadas hacia la prevalencia de entrenamiento, y por ello requieren calibración post-entrenamiento.

---

## 2. Método de calibración por modelo

La tabla de calibración está definida en `ALGORITHM_CALIBRATION_METHODS` dentro de [`src/models/registry.py`](../../src/models/registry.py):

| Modelo | Clase sklearn/xgboost | Método de calibración | Justificación |
|---|---|---|---|
| `logistic_regression` | `LogisticRegression` | **sigmoid** (Platt scaling) | La salida ya es ~lineal en log-odds; Platt es suficiente y más estable |
| `mlp` | `MLPClassifier` | **sigmoid** (Platt scaling) | La activación final (sigmoid/softmax) produce salida ~lineal en log-odds |
| `random_forest` | `RandomForestClassifier` | **isotonic** | Los votos promedio del RF tienden a concentrarse en rangos; isotónica más flexible |
| `extra_trees` | `ExtraTreesClassifier` | **isotonic** | Igual que RF — alta varianza en probabilidades crudas |
| `xgboost` | `XGBClassifier` | **isotonic** | Boosting produce probabilidades sesgadas hacia la prevalencia; isotónica mejor |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | **isotonic** | Mismo motivo que XGBoost |
| `lightgbm` | `LGBMClassifier` | **isotonic** | Mismo motivo |
| `stacking` | `StackingClassifier` | **isotonic** | El meta-estimador recibe probabilidades de base estimators; isotónica más robusta |
| `voting` | `VotingClassifier` | **isotonic** | Promedio de probabilidades puede concentrarse; isotónica corrige |

La regla de selección sigue dos criterios. El método **sigmoid (Platt scaling)** es paramétrico: entrena una función logística sobre las probabilidades crudas y funciona bien cuando la salida del modelo es aproximadamente lineal en log-odds. La **regresión isotónica** es no paramétrica: ajusta una función monótonamente creciente, más flexible para distribuciones arbitrarias, aunque requiere un volumen de muestras suficiente para ser estable (con pocos datos puede sobreajustarse).

Si el método primario falla durante el entrenamiento, `fit_with_calibration` recurre automáticamente al método alternativo. Si ambos fallan, el modelo se persiste sin calibrar y se registra una advertencia en el campo `calibration_info["warnings"]` del manifest.

---

## 3. Implementación

### 3.1 Calibración durante el entrenamiento (`fit_with_calibration`)

La calibración se aplica **durante el entrenamiento** en `src/models/trainer.py`, llamando a `fit_with_calibration` de `src/models/registry.py`. El flujo es:

```
1. Split interno 80/20 estratificado (train_inner / val_inner)
2. fit_with_calibration(model_cfg, X_train_inner, y_train_inner)
   ├── Construye el estimador base (_build_base_model)
   ├── Lo envuelve en CalibratedClassifierCV(method, cv=cv)
   └── Entrena el conjunto (base + calibrador) con cross-validation interna
3. Busca threshold óptimo sobre val_inner con probabilidades ya calibradas
4. Re-entrena el modelo final en X completo (también con fit_with_calibration)
5. Persiste el modelo calibrado como {model_name}_model.joblib
```

El parámetro de cross-validation interna para la calibración se toma de `calibration_cv` en `models_config.yaml` (por defecto: `cv=5`). Todos los modelos tienen `calibrate: true` en la config.

```python
# src/models/registry.py — función central
def fit_with_calibration(model_cfg, X, y, random_state=42):
    primary_method = _resolve_calibration_method(class_name, override=...)
    cv = int(model_cfg.get("calibration_cv", 5))
    # Intenta primary_method → fallback a alternativo → sin calibración
    wrapped = CalibratedClassifierCV(base_model, method=primary_method, cv=cv)
    wrapped.fit(X, y)
    return wrapped, {"calibrated": True, "method": primary_method, "cv": cv, ...}
```

El estado de calibración se registra en el manifest del modelo bajo la clave `calibration`:

```json
{
  "calibration": {
    "calibrated": true,
    "method": "isotonic",
    "cv": 5,
    "fallback_reason": null,
    "warnings": []
  }
}
```

### 3.2 Tarea de calibración en el DAG (`make_task_calibration`)

Después de la tarea `evaluate`, el DAG ejecuta `make_task_calibration` para **medir la calidad** de la calibración sobre el test set. La tarea:

1. Carga el modelo serializado (`.joblib`) y el test set.
2. Obtiene `y_proba = model.predict_proba(X_test_num)[:, 1]`.
3. Llama a `compute_calibration_curve(y_test, y_proba)` de `src/evaluation/calibration.py`.
4. Escribe el JSON de calibración en `cal_dir / f"calibration_{model_key}.json"`.
5. Genera el reliability diagram PNG con `plot_calibration_curve(...)`.

```python
# dags/preanesthesia_pipeline.py — make_task_calibration (simplificado)
cal_data = compute_calibration_curve(y_test.values, y_proba)
(cal_dir / f"calibration_{model_key}.json").write_text(json.dumps(cal_data, indent=2))
plot_calibration_curve(cal_data, model_key, target_name, cal_dir)
```

Dependencias en el DAG:
```
evaluate__{modelo}__{target}
        │
        ▼
calibration__{modelo}__{target}
        │
        ▼
calibration_comparison__{target}   ← agrega todos los modelos en un único plot
```

### 3.3 Parámetros de `compute_calibration_curve`

| Parámetro | Valor usado | Descripción |
|---|---|---|
| `n_bins` | `10` | 10 bins de igual ancho (resolución estándar clínica) |
| `strategy` | `"uniform"` | Bins de ancho igual en [0, 1] — no por cuantiles |

---

## 4. Métricas de evaluación de calibración

`compute_calibration_curve` retorna un diccionario con las siguientes métricas:

| Campo | Tipo | Descripción |
|---|---|---|
| `fraction_of_positives` | array | Proporción real de positivos en cada bin — eje Y del reliability diagram |
| `mean_predicted` | array | Probabilidad media predicha en cada bin — eje X del reliability diagram |
| `bin_counts` | array | Número de muestras por bin — permite ponderar el ECE y estimar confianza por bin |
| `ece` | float | **Expected Calibration Error** — error de calibración medio ponderado por tamaño de bin. ECE=0 es calibración perfecta; valores <0.05 son considerados buenos |
| `mce` | float | **Maximum Calibration Error** — error máximo en el bin más desviado. Útil para detectar bins problemáticos aunque el ECE global sea bajo |
| `brier` | float | **Brier Score** — MSE entre probabilidad predicha y etiqueta real. Combina discriminación y calibración; menor es mejor |
| `overconfident_bins` | int | Número de bins donde `mean_predicted > fraction_of_positives` (el modelo sobreestima el riesgo) |
| `underconfident_bins` | int | Número de bins donde `mean_predicted < fraction_of_positives` (el modelo subestima el riesgo) |
| `n_bins` | int | Número de bins usados (10) |
| `strategy` | str | Estrategia de bins (`"uniform"`) |
| `n_samples` | int | Total de muestras en el test set |

**Interpretación de ECE:**
- ECE < 0.02: calibración excelente
- ECE 0.02–0.05: calibración aceptable para uso clínico
- ECE > 0.05: calibración deficiente — las probabilidades no deben interpretarse como riesgos reales sin corrección adicional

---

## 5. Ejemplo de reporte real

`output/v2/reports/calibration/target_d_v2_hosp/calibration_logistic_regression.json`:

```json
{
  "fraction_of_positives": [
    0.09051724137931035,
    0.16875460574797346,
    0.24526928675400292,
    0.32068965517241377,
    0.43232323232323233,
    0.51010101010101,
    0.7428571428571429,
    0.7804878048780488,
    0.8,
    0.0
  ],
  "mean_predicted": [
    0.07940421881129128,
    0.15533837862652253,
    0.2470333287027056,
    0.3438503637568421,
    0.4453214867381704,
    0.5428553435124402,
    0.6452914422775455,
    0.7342348938605867,
    0.8356013765545729,
    0.9618926217669536
  ],
  "bin_counts": [232, 1357, 1374, 870, 495, 198, 105, 41, 5, 1],
  "ece": 0.014869,
  "mce": 0.961893,
  "brier": 0.182446,
  "overconfident_bins": 6,
  "underconfident_bins": 4,
  "n_bins": 10,
  "strategy": "uniform",
  "n_samples": 4678
}
```

El `ece=0.0149` confirma que la Regresión Logística está muy bien calibrada: el error promedio ponderado es inferior al 1.5%. El `mce=0.9619` corresponde al último bin (probabilidades predichas ~0.96), que contiene una única muestra con fracción de positivos de 0; este MCE elevado no es representativo dado que el bin está prácticamente vacío. El `brier=0.1824` es razonable para un problema con 27.7% de prevalencia. Los `overconfident_bins=6` indican que en seis de los diez bins el modelo sobreestima el riesgo, patrón típico en clasificadores entrenados con `class_weight="balanced"`. La concentración de la masa de predicciones en los bins de probabilidad baja (`bin_counts=[232, 1357, 1374, ...]`) es consistente con un umbral operativo de 0.13–0.18.

---

## 6. Plots generados

### 6.1 Reliability diagram individual

Generado por `plot_calibration_curve` de `src/reports/calibration_plots.py`. Archivo: `output/v2/reports/calibration/{target}/calibration_{modelo}.png`.

El gráfico contiene dos paneles. El panel superior muestra el reliability diagram con la diagonal de calibración perfecta (línea discontinua negra), los puntos de calibración real (tamaño proporcional al número de muestras del bin, color por probabilidad predicha) y las regiones sombreadas de sobreconfianza (rojo) y subconfianza (azul); el título incluye los valores de ECE y Brier. El panel inferior presenta el histograma de la distribución de probabilidades predichas, permitiendo identificar dónde se concentran las predicciones del modelo.

### 6.2 Reliability diagram comparativo

Generado por `make_task_calibration_comparison` del DAG, que agrega todos los JSONs de calibración del target. Archivo: `output/v2/reports/calibration/{target}/calibration_comparison.png`.

Presenta todos los modelos superpuestos en un único reliability diagram, acompañado de una tabla resumen con ECE, Brier, bins sobreconfiados y subconfiados, ordenada por ECE ascendente. El modelo con mejor calibración aparece resaltado en verde.

**Archivos de plots disponibles en el repositorio:**

Para `target_d_v2_hosp`:
- `output/v2/reports/calibration/target_d_v2_hosp/calibration_logistic_regression.png`
- `output/v2/reports/calibration/target_d_v2_hosp/calibration_random_forest.png`
- `output/v2/reports/calibration/target_d_v2_hosp/calibration_extra_trees.png`

Para `target_f_predictibilidad_maxima`:
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_logistic_regression.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_random_forest.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_extra_trees.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_xgboost.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_hist_gradient_boosting.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_lightgbm.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_mlp.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_stacking.png`
- `output/v2/reports/calibration/target_f_predictibilidad_maxima/calibration_comparison.png`
