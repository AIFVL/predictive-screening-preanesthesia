# Etapa 3 — Selección de Features

**Código fuente:**
- [`src/features/selection.py`](../../src/features/selection.py) — Rankeo y selección por MI + RF
- [`src/features/engineering.py`](../../src/features/engineering.py) — Inferencia de features candidatas, fixes de encoding
- [`src/reports/pre_post_analysis.py`](../../src/reports/pre_post_analysis.py) — Análisis de correlación preop→target
- [`src/reports/correlation.py`](../../src/reports/correlation.py) — Matrices de correlación

**Outputs (pipeline v2):**
- [`output/v2/data_processed/target_d_v2_hosp/selected_features.json`](../../output/v2/data_processed/target_d_v2_hosp/selected_features.json) — Features seleccionadas para `target_d_v2_hosp`
- [`output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json`](../../output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json) — Features seleccionadas para `target_f_predictibilidad_maxima`
- [`output/v2/reports/pre_post_signal/pre_post_linkage_per_flag.csv`](../../output/v2/reports/pre_post_signal/pre_post_linkage_per_flag.csv) — MI y correlación por feature
- [`output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv`](../../output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv) — Resumen por versión del target

**Configuración:**
- [`config/features_config.yaml`](../../config/features_config.yaml) — Umbral de varianza mínima, parámetros del RF de rankeo, threshold del score combinado.

---

## 1. El problema de partida: 236 features, señal débil

Tras la limpieza, el dataset preoperatorio contiene **236 columnas**. No todas contribuyen de igual forma a la predicción del target. Existen tres categorías de problemas: columnas con varianza casi nula, donde el 99% de los pacientes comparten el mismo valor y no aportan poder discriminativo; columnas numéricamente válidas pero sin correlación real con el target, que introducen ruido y pueden favorecer el sobreajuste; y columnas altamente correlacionadas entre sí, como `score_proc_high_severity` y `score_proc_critical`, que capturan conceptos solapados. El objetivo de esta etapa es **reducir las 236 features a un subconjunto informativo y manejable** que conserve la mayor parte de la señal predictiva disponible.

---

## 2. Metodología de selección

### 2.1 Poda de baja varianza

Antes del rankeo formal, se eliminan columnas con varianza nula. La función `sanitize_features` en `src/features/selection.py` descarta las columnas donde `nunique() <= 1`, es decir, columnas con un único valor posible (varianza exactamente cero). El parámetro `min_variance: 0.01` está declarado en `config/features_config.yaml` bajo `feature_pruning`, pero actualmente **no está activo en el código de selección** — es una configuración declarada, no implementada. El filtro real aplicado es solo el de varianza exactamente cero.

### 2.2 Inferencia de features candidatas

La función `infer_candidate_features()` en `src/features/engineering.py` identifica automáticamente qué columnas del `merged.parquet` son candidatas para el modelo, excluyendo:
- Columnas identificadoras (`Documento PMD`, `CODIGO`)
- Columnas que son el target o subflags del target (`target`, `n_flags_relevant`)
- Columnas no numéricas que no pudieron ser codificadas

### 2.3 Score combinado MI + Random Forest

El rankeo principal usa **dos métricas independientes** que se combinan:

#### Mutual Information (MI)
La Información Mutua entre cada feature y el target cuantifica cuánta información sobre el target se puede obtener conociendo el valor de esa feature. A diferencia de la correlación de Pearson, captura relaciones no lineales y no impone supuestos distribucionales.

```python
mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
```

Valores MI más altos indican mayor poder predictivo de la feature sobre el target. Su principal limitación es la tendencia a sobrevalorar features de alta cardinalidad y a producir estimaciones ruidosas con pocos datos.

#### Feature Importance de Random Forest
Se entrena un Random Forest específicamente para el proceso de rankeo —distinto del modelo final— y se utiliza la importancia media de disminución de impureza (MDI) de cada feature como medida de relevancia:

```python
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_split=20,
    min_samples_leaf=10, class_weight="balanced",
)
rf.fit(X, y)
importances = rf.feature_importances_
```

A diferencia de MI, la importancia de Random Forest captura interacciones entre features.

#### Score combinado

Ambas métricas se normalizan a [0,1] con MinMaxScaler y se promedian:

```
Combined_Score = (MI_Norm + RF_Norm) / 2
```

Se seleccionan las features con `Combined_Score ≥ 0.02`. Este umbral retiene aproximadamente las 80 features más informativas y descarta las que tienen score casi nulo en ambas métricas.

---

## 3. Análisis de señal preop→target

Antes de entrenar los modelos, se calculó el enlace estadístico entre cada feature preoperatoria y el target usando:
- **Mutual Information** — para relaciones no lineales
- **Correlación de Pearson (valor absoluto)** — para relaciones lineales

Los resultados están en `output/v2/reports/pre_post_signal/pre_post_linkage_per_flag.csv`.

### Hallazgos del análisis de señal

Resumen exacto extraído de [`output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv`](../../output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv):

| Versión target | Prevalencia | Max MI | Max Pearson | N features informativas | Top 3 features preop |
|---|---|---|---|---|---|
| `target_d_v2_hosp` | 27.69% | 0.10032 | 0.23227 | 16 | `score_proc_high_severity`, `score_proc_moderate_severity`, `score_proc_medium_severity` |
| `target_f_predictibilidad_maxima` | 19.43% | **0.12972** | **0.30335** | 16 | `score_proc_moderate_severity`, `score_proc_high_severity`, `score_proc_low_severity` |

`target_f_predictibilidad_maxima` exhibe aproximadamente un 30% más de MI máxima y un 31% más de correlación de Pearson que `target_d_v2_hosp`, a pesar de tener una prevalencia menor (19.43% vs. 27.69%). Esto es consecuencia directa de su construcción: se compone exclusivamente de flags identificados por el Enfoque C como los más predecibles desde variables preoperatorias. La conclusión metodológica es clara: la calidad del target —qué flags se incluyen— tiene mayor impacto sobre la señal disponible que su amplitud —cuántos flags se incluyen. Un target compuesto por 5 flags altamente predecibles supera a uno formado por 7 flags de los cuales algunos introducen ruido.

**Top features por señal para `target_d_v2_hosp`:**
1. `score_proc_high_severity` — MI: 0.100, Pearson: 0.232
2. `score_proc_moderate_severity` — MI alto, Pearson moderado
3. `score_proc_medium_severity` — MI alto, Pearson moderado

**Top features por señal para `target_f_predictibilidad_maxima`:**
1. `score_proc_moderate_severity` — Pearson: 0.30
2. `score_proc_high_severity` — Pearson alto
3. `score_proc_low_severity` — Pearson moderado

Los scores de severidad del procedimiento dominan la señal en ambos targets. **La complejidad del procedimiento es el predictor más fuerte del target**, más que las comorbilidades del paciente.

---

## 4. Las features seleccionadas

> **Nota sobre tamaños:** El número exacto de features seleccionadas depende del target. El listado de 80 features mostrado a continuación corresponde a `target_d_v2_hosp` (versión histórica). El listado de `target_f_predictibilidad_maxima` (servido por la API) contiene 59 features y se documenta más abajo en la sección 4.3.

### 4.1 Distribución por tipo (`target_d_v2_hosp`)

De las 236 features originales, **80 superaron el umbral de Combined_Score ≥ 0.02**:

| Tipo de feature | N features | Ejemplos |
|----------------|-----------|---------|
| Scores de severidad (proc/dx/ant) | ~15 | `score_proc_high_severity`, `score_dx_critical`, ... |
| Tipo de anestesia propuesta (OHE) | 7 | `_raquidea`, `_general`, `_peridural`, ... |
| Capítulo CIE-10 del diagnóstico (OHE) | ~15 | `Dx Preoperatorio Code_O`, `_S`, `_H`, ... |
| Código de procedimiento propuesto (OHE) | ~8 | `Procedimiento propuesto Code_L`, ... |
| Antecedentes médicos | ~10 | `hta`, `diabetes`, `negativo`, ... |
| Variables numéricas de examen físico | ~10 | `Edad`, `IMC`, `TA Sistólica`, `Hemoglobina`, ... |
| Variables temporales | 4 | `anio`, `mes`, `dia_semana`, `Hora_decimal` |
| Otros | ~11 | `Sexo_encoded`, `Puntaje Mallampati`, `Estado Nutricional_encoded`, ... |

### 4.2 Las 80 features de `target_d_v2_hosp` ordenadas por ranking descendente de Combined_Score

1. `score_proc_high_severity` — Procedimiento de alta severidad ⭐ (predictor #1)
2. `score_proc_critical` — Procedimiento crítico
3. `Tipo de anestesia propuesta_raquidea` — Anestesia raquídea
4. `score_proc_low_severity` — Procedimiento de baja severidad
5. `score_proc_medium_severity` — Procedimiento de severidad media
6. `score_proc_moderate_severity` — Procedimiento de severidad moderada
7. `score_dx_high_severity` — Diagnóstico de alta severidad
8. `Tipo de anestesia propuesta_sedacion` — Anestesia con sedación
9. `score_dx_critical` — Diagnóstico crítico
10. `score_dx_medium_severity` — Diagnóstico de severidad media
11. `score_dx_low_severity` — Diagnóstico de baja severidad
12. `score_dx_moderate_severity` — Diagnóstico de severidad moderada
13. `Tipo de anestesia propuesta_peridural` — Anestesia peridural
14. `Tipo de anestesia propuesta_local` — Anestesia local
15. `Dx Preoperatorio Code_O` — Diagnóstico cap. O (embarazo/parto)
16. `Examen_Hemoglobina(g/dl)` — Hemoglobina
17. `Edad` — Edad del paciente
18. `Dx Preoperatorio Code_S` — Diagnóstico cap. S (traumatismos)
19. `IMC` — Índice de masa corporal
20. `Tipo de anestesia propuesta_general` — Anestesia general
21. `Procedimiento propuesto Code_L` — Procedimiento cap. L (piel)
22. `Hora_decimal` — Hora de la valoración
23. `score_proc_ant_low_severity` — Antecedente quirúrgico de baja severidad
24. `Tensión Arterial Media (mm/Hg)` — Presión arterial media basal
25. `Talla (cm)` — Talla del paciente
26. `score_proc_ant_moderate_severity` — Antecedente quirúrgico severidad moderada
27. `Dx Preoperatorio Code_A` — Diagnóstico cap. A (infecciosas)
28. `Peso (Kg)` — Peso del paciente
29. `Antecedente renales_negativo` — Sin antecedentes renales
30. `score_proc_ant_high_severity` — Antecedente quirúrgico de alta severidad
31. `score_proc_ant_critical` — Antecedente quirúrgico crítico
32. `Temperatura` — Temperatura basal
33. `Frecuencia Respiratoria` — Frecuencia respiratoria basal
34. `Dx Preoperatorio Code_Z` — Diagnóstico cap. Z (factores de salud)
35. `Tensión Arterial Sistólica (mm/Hg)` — TA sistólica basal
36. `Antecedentes anestésicos` — Antecedentes anestésicos previos
37. `score_proc_ant_medium_severity` — Antecedente quirúrgico severidad media
38. `Sexo_encoded` — Sexo del paciente (codificado)
39. `Tensión Arterial Diastólica (mm/Hg)` — TA diastólica basal
40. `Grupo Sanguíneo_Sin Dato` — Grupo sanguíneo desconocido
41. `anio` — Año del registro
42. `Procedimiento propuesto Code_NO ENCONTRADO` — Procedimiento no clasificado
43. `Antecedente hematológicos _negativo` — Sin antecedentes hematológicos
44. `Grupo Sanguíneo_O` — Grupo sanguíneo O
45. `mes` — Mes del registro
46. `Dx Preoperatorio Code_H` — Diagnóstico cap. H (ojo/oído)
47. `predicted_label_proc_encoded` — Clasificación del procedimiento (LLM)
48. `Estado Nutricional_encoded` — Estado nutricional (normal, sobrepeso, obeso, ...)
49. `Grupo Sanguíneo_A` — Grupo sanguíneo A
50. `Antecedente endocrinológicos_negativo` — Sin antecedentes endocrinológicos
51. `Dx Preoperatorio Code_M` — Diagnóstico cap. M (musculoesquelético)
52. `Sistema Respiratorio_disnea` — Disnea en examen físico
53. `Anestesia previa_sedacion` — Anestesia previa: sedación
54. `predicted_label_dx_encoded` — Clasificación del diagnóstico (LLM)
55. `Grupo Sanguíneo_B` — Grupo sanguíneo B
56. `Antecedente gastrointestinales_gastritis` — Antecedente de gastritis
57. `Antecedentes quirúrgicos Code_L` — Antecedente quirúrgico cap. L
58. `Puntaje Mallampati` — Clasificación de vía aérea
59. `Prótesis Dental_movil` — Prótesis dental móvil
60. `Sistema cardiovascular_normal` — Sistema cardiovascular normal en examen
61. `Antecedente neurológicos_neoplasia` — Neoplasia neurológica
62. `Tipo de anestesia propuesta_sin dato` — Sin dato de tipo de anestesia
63. `Dx Preoperatorio Code_G` — Diagnóstico cap. G (nervioso)
64. `Antecedentes cardiovasculares_hta` — Hipertensión arterial
65. `dia_semana` — Día de la semana
66. `RH_Sin Dato` — Factor RH desconocido
67. `Procedimiento propuesto Code_S` — Procedimiento cap. S (traumatismos)
68. `Sistema Respiratorio_normal` — Sistema respiratorio normal en examen
69. `Procedimiento propuesto Code_C` — Procedimiento cap. C (neoplasias)
70. `Anestesia previa_neuroaxial` — Anestesia previa neuroaxial
71. `Alérgeno_med_opioides` — Alergia a opioides
72. `Alérgeno_med_neuro_psiquiatria` — Alergia a fármacos neuropsi
73. `Antecedentes cardiovasculares_negativo` — Sin antecedentes cardiovasculares
74. `Procedimiento propuesto Code_M` — Procedimiento cap. M (musculoesquelético)
75. `Respuesta Motora_encoded` — Respuesta motora en escala de Glasgow
76. `Examen_PT (INR)` — INR de coagulación
77. `Sistema Nervioso_Sin dato` — Sin dato de sistema nervioso
78. `Prótesis Dental_no` — Sin prótesis dental
79. `Antecedente renales_litiasis` — Litiasis renal
80. `Alérgeno_med_antiinfecciosos` — Alergia a antibióticos

### 4.3 Las 59 features de `target_f_predictibilidad_maxima`

Para `target_f_predictibilidad_maxima` se seleccionan **59 features** (extraídas de [`output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json`](../../output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json) y del manifest de XGBoost). El conjunto es muy similar al de `target_d_v2_hosp` salvo que descarta features con baja relación con los flags de UCI/hospitalización.

Top features para `target_f`:

1. `Tipo de anestesia propuesta_raquidea`
2. `score_proc_high_severity`
3. `score_proc_low_severity`
4. `score_proc_medium_severity`
5. `score_proc_moderate_severity`
6. `score_proc_critical`
7. `score_dx_low_severity`
8. `score_dx_critical`
9. `score_dx_medium_severity`
10. `score_dx_high_severity`
11. `score_dx_moderate_severity`
12. `Tipo de anestesia propuesta_sedacion`
13. `Tipo de anestesia propuesta_peridural`
14. `Tipo de anestesia propuesta_local`
15. `Dx Preoperatorio Code_O`
16. `Dx Preoperatorio Code_S`
17. `Examen_Hemoglobina(g/dl)`
18. `Tipo de anestesia propuesta_general`
19. `Edad`
20. `IMC`
21. … *(continúa hasta 59 features)*

Listado completo en [`output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json`](../../output/v2/data_processed/target_f_predictibilidad_maxima/selected_features.json).

**Diferencias clave con el conjunto de `target_d_v2_hosp`:**
- El orden es similar pero `Tipo de anestesia propuesta_raquidea` ocupa el primer puesto del ranking nativo (no el `score_proc_high_severity`).
- Aparecen `Dx Preoperatorio Code_K` y `Procedimiento propuesto Code_A` que no estaban en el top de `target_d_v2_hosp`.
- Se excluyen algunos antecedentes de comorbilidades de baja MI con los flags de UCI/hospitalización.

### 4.4 Interpretación del ranking

Las primeras 14 features del ranking corresponden a scores de severidad y tipo de anestesia propuesta — todas derivadas de la complejidad del procedimiento y el diagnóstico, no del perfil clínico del paciente. El modelo predice principalmente en función de **qué se va a hacer** más que de **quién es el paciente**.

Las comorbilidades aparecen tarde. `Antecedentes cardiovasculares_hta` se sitúa en la posición 64; `Antecedente endocrinológicos_diabetes` queda por debajo del umbral de selección, aunque el análisis de subgrupos (Enfoque B) muestra que los pacientes diabéticos son precisamente los que el modelo predice con mayor dificultad. Esto es coherente: si diabetes tiene baja MI con el target actual, el modelo no incorpora esa señal y, en consecuencia, falla sistemáticamente en pacientes endocrinológicos.

Las variables del examen físico — `Edad`, `IMC`, `TA Sistólica`, `Hemoglobina`, `Temperatura`, `Frecuencia Respiratoria` — se ubican en el tercio medio del ranking. Son relevantes, pero no determinantes. Las variables temporales (`anio`, `mes`, `dia_semana`, `Hora_decimal`) superan el umbral mínimo y capturan una tendencia temporal moderada en los datos, posiblemente relacionada con cambios en protocolos o en el perfil de los pacientes atendidos en distintos períodos.

---

## 5. Diagnóstico principal: el modelo predice complejidad del procedimiento más que riesgo del paciente

El análisis de features revela un patrón con implicaciones directas sobre la utilidad clínica del modelo. Los predictores más importantes — `score_proc_*`, `score_dx_*` y el tipo de anestesia propuesta — están todos relacionados con la complejidad del procedimiento y el diagnóstico principal, no con el perfil clínico del paciente. Estos factores son informativos, pero también los más evidentes: cualquier médico sabe que un procedimiento de alta complejidad conlleva mayor riesgo de complicaciones.

Las comorbilidades del paciente — precisamente lo que una valoración preanestésica formal evalúa — tienen señal débil en el modelo actual. Diabetes, HTA, EPOC e insuficiencia renal no figuran entre los predictores dominantes. Esto sugiere que el modelo aprende principalmente la asociación "procedimiento complejo → complicación probable" en lugar de "paciente de alto riesgo médico → requiere valoración formal". Esta distinción es relevante porque la complejidad del procedimiento es información disponible sin necesidad de un modelo de ML.

El valor añadido real de la valoración preanestésica reside en identificar pacientes con factores de riesgo no evidentes a priori: el diabético con control metabólico deficiente, el hipertenso con función renal comprometida, el obeso con apnea del sueño no diagnosticada. Estos son precisamente los pacientes para quienes una valoración formal modifica el manejo clínico y, a la vez, los que el modelo actual detecta con menor fiabilidad. Este diagnóstico es coherente con los hallazgos del Enfoque B — el modelo falla más en pacientes endocrinológicos y cardiovasculares — y del Enfoque C, que muestra que las comorbilidades crónicas son predictores débiles para la mayoría de los flags posoperatorios.

---

## 6. Señal disponible en perspectiva

Comparativa final entre las versiones del target activas en v2 (más una versión histórica para referencia):

| Versión target | Prevalencia | Max MI | Max Pearson | N features informativas | AUC mejor modelo |
|---|---|---|---|---|---|
| `target_d_v2` *(legacy)* | 16.93% | 0.031 | 0.154 | 11 | ~0.64 |
| `target_d_v2_hosp` | 27.69% | 0.100 | 0.232 | 16 | ~0.76 |
| `target_f_predictibilidad_maxima` *(recomendado)* | 19.43% | **0.130** | **0.303** | 16 | **~0.86** |

El salto en la correlación de Pearson de 0.23 a 0.30 al cambiar de `target_d_v2_hosp` a `target_f_predictibilidad_maxima` es estadísticamente significativo: cada feature aporta más información discriminativa, y el techo del modelo sube a AUC ~0.86 en lugar de converger en ~0.76. Para referencia, una feature con Pearson = 1.0 predice el target de forma perfecta; con 0.23 existe información real pero es necesario combinar muchas features para extraerla; con 0.30, las combinaciones tienen mayor capacidad discriminativa nativa.

Este resultado confirma el principio rector del proyecto: **el techo de rendimiento está determinado por la señal disponible en los datos, que es función de la definición del target, no del algoritmo elegido**. La selección del target tuvo un impacto en el AUC de mayor magnitud que cualquier decisión algorítmica, de regularización o de ajuste de hiperparámetros.
