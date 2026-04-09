# Pipeline de Modelado — Documentación Completa

> **Proyecto:** Screening predictivo de valoración preanestésica — Fundación Valle del Lili  
> **Rama git:** `feat/refactor-pipeline`  
> **Fecha de documentación:** 2026-04-05

---

## Objetivo del proyecto

Desarrollar un modelo de clasificación que prediga qué pacientes, al momento de la valoración preanestésica, necesitan una evaluación formal adicional antes de entrar a cirugía. El modelo se alimenta exclusivamente de variables registradas durante la consulta preanestésica (variables preoperatorias) para predecir si el paciente presentará complicaciones o eventos adversos posoperatorios.

Este es un problema de **clasificación binaria desbalanceada** en un contexto médico donde el coste de un falso negativo (no detectar a un paciente que sí necesita evaluación) es mucho mayor que el de un falso positivo (enviar a evaluación a un paciente que no la necesita). Esto determina todas las decisiones de umbral y métrica del proyecto.

---

## Estructura del pipeline

```
Datos brutos (Excel)
        │
        ▼
[1. Limpieza y preprocesamiento]   → output/v1/data_processed/preop_raw.parquet
        │                          → output/v1/data_processed/posop_raw.parquet
        │                          → output/v1/data_processed/cleaned.parquet
        │                          → output/v1/reports/cleaning_report.json
        │
        ▼
[2. Construcción del target]       → posop_raw.parquet + columna "target"
        │                          → Múltiples versiones: target_d_v2, target_d_v2_hosp, target_d_v5
        │
        ▼
[3. Join preop + target]           → output/v1/data_processed/{target}/merged.parquet
        │                          → 23,387 pacientes con variables preop + target
        │
        ▼
[4. Selección de features]         → output/v1/data_processed/{target}/selected_features.json
        │                          → 80 features seleccionadas de 236 candidatas
        │
        ▼
[5. División train/test]           → output/v1/data_processed/{target}/splits/
        │                          → 80/20, estratificado por target
        │
        ▼
[6. Entrenamiento de modelos]      → output/v1/models/{target}/{modelo}_model.joblib
        │                          → output/v1/models/{target}/{modelo}_metrics.json
        │
        ▼
[7. Evaluación y explicabilidad]   → output/v1/reports/explainability/{target}/
                                   → output/v1/plots/{target}/
```

---

## Documentos disponibles

| Etapa | Documento |
|-------|-----------|
| Limpieza y preprocesamiento | [01-limpieza-datos.md](01-limpieza-datos.md) |
| Construcción del target | [02-construccion-target.md](02-construccion-target.md) |
| Selección de features | [03-seleccion-features.md](03-seleccion-features.md) |
| Entrenamiento y evaluación | [04-entrenamiento-evaluacion.md](04-entrenamiento-evaluacion.md) |
| Explicabilidad | [05-explicabilidad.md](05-explicabilidad.md) |
| Análisis exploratorio posoperatorio | [../analisis-posoperatorio/README.md](../analisis-posoperatorio/README.md) |

---

## Fuentes de datos

El proyecto usa dos datasets principales que provienen del sistema de información hospitalario de la Fundación Valle del Lili:

### Dataset preoperatorio (valoración preanestésica)
- **Origen:** Registros de consultas preanestésicas del sistema OPERA
- **Archivo bruto:** Excel / formato hospitalario
- **Procesado:** `output/v1/data_processed/preop_raw.parquet`
- **Filas originales:** 30,962 registros
- **Filas tras filtrado (≥18 años):** 24,279 registros
- **Columnas originales:** 236 variables
- **Periodo:** Múltiples años de registros históricos
- **Clave de unión:** `Documento PMD` (identificador único de episodio)

### Dataset posoperatorio (registro de quirófano)
- **Origen:** Registros posoperatorios del sistema OPERA (hoja de anestesia)
- **Archivo bruto:** Excel / formato hospitalario
- **Procesado:** `output/v1/data_processed/posop_raw.parquet`
- **Filas:** 29,865 registros
- **Columnas:** 134 variables + 57 flags derivados
- **Clave de unión:** `Documento PMD (valoración preanestésica)`

### Join entre datasets
- **Tipo:** INNER JOIN por `Documento PMD`
- **Resultado:** 23,387 pacientes con datos completos de ambas fuentes
- **Pacientes preop sin match posop:** 892 (excluidos del modelado)

---

## Flujo de datos numérico

```
30,962  registros preop brutos
 - 6,683  excluidos por edad < 18 años
= 24,279  en cleaned.parquet

24,279  registros limpios
× INNER JOIN con 29,865 posop
= 23,387  en merged.parquet (por Documento PMD)

23,387  en merged
 - 892   sin registro posop (diferencia preop - merged)

23,387  con target
├── 6,475  positivos (27.7%) — necesitan valoración formal
└── 16,912 negativos (72.3%)

División 80/20 estratificada:
├── Train: 18,709 (5,180 positivos, 27.7%)
└── Test:   4,678 (1,295 positivos, 27.7%)
```

---

## Decisiones de diseño transversales

### 1. Métrica principal: F2 y Recall
En un contexto médico de screening, la consecuencia de un **falso negativo** (no identificar a un paciente que sí necesita valoración) es potencialmente grave. Por eso se usa **F2-score** como métrica de optimización — da el doble de peso al Recall que a la Precisión. El Recall se monitoriza directamente como métrica secundaria.

### 2. Thresholds bajos (0.17)
Los modelos se operan con thresholds muy bajos (0.17 en los mejores modelos), lo que significa que clasifican como positivo a cualquier paciente para el que el modelo asigna ≥17% de probabilidad de complicación. Esto resulta en una tasa de predicción positiva del 67.3% — es decir, el modelo dice "este paciente necesita valoración" para 2 de cada 3 pacientes. El tradeoff consciente es: se aceptan muchos falsos positivos a cambio de minimizar falsos negativos.

### 3. Uso de `class_weight="balanced"`
Los modelos de árbol y de bosque se entrenan con `class_weight="balanced"` para compensar el desbalance de clases (72.3% negativos vs. 27.7% positivos). Esto hace que el modelo pese más los errores en positivos durante el entrenamiento.

### 4. Calibración de probabilidades
Los modelos basados en árboles se calibran con `CalibratedClassifierCV` (isotonic) sobre el training set. Esto convierte los scores crudos del modelo (que no son probabilidades bien calibradas) en estimaciones de probabilidad más confiables, mejorando la utilidad del threshold.

### 5. Múltiples versiones del target
Se entrenaron tres versiones del target para comparar cuál produce el modelo más predecible. La versión `target_d_v2_hosp` resultó la mejor por ROC AUC y fue seleccionada para análisis posterior.
