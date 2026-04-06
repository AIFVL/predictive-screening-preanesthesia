# Etapa 3 — Selección de Features

**Código fuente:**
- [`src/features/selection.py`](../../src/features/selection.py) — Rankeo y selección por MI + RF
- [`src/features/engineering.py`](../../src/features/engineering.py) — Inferencia de features candidatas, fixes de encoding
- [`src/reports/pre_post_analysis.py`](../../src/reports/pre_post_analysis.py) — Análisis de correlación preop→target
- [`src/reports/correlation.py`](../../src/reports/correlation.py) — Matrices de correlación

**Outputs:**
- [`output/v1/data_processed/target_d_v2_hosp/selected_features.json`](../../output/v1/data_processed/target_d_v2_hosp/selected_features.json) — 80 features seleccionadas
- [`output/v1/reports/pre_post_signal/pre_post_linkage_per_flag.csv`](../../output/v1/reports/pre_post_signal/pre_post_linkage_per_flag.csv) — MI y correlación por feature
- [`output/v1/reports/pre_post_signal/pre_post_linkage_summary.csv`](../../output/v1/reports/pre_post_signal/pre_post_linkage_summary.csv) — Resumen por versión del target

---

## 1. El problema de partida: 236 features, señal débil

Tras la limpieza, el dataset preoperatorio tiene **236 columnas**. No todas son igualmente útiles para predecir el target. Hay tres tipos de problemas:

1. **Features con varianza casi nula:** Columnas donde el 99% de los pacientes tienen el mismo valor. No añaden información discriminativa.
2. **Features ruidosas:** Columnas numéricamente válidas pero sin correlación real con el target — añaden ruido y pueden causar sobreajuste.
3. **Features redundantes:** Columnas altamente correlacionadas entre sí (por ejemplo, `score_proc_high_severity` y `score_proc_critical` capturan conceptos solapados).

El objetivo de la selección es **reducir las 236 features a un subconjunto informativo y manejable**, manteniendo la mayor parte de la señal predictiva.

---

## 2. Metodología de selección

### 2.1 Poda de baja varianza

Antes del rankeo formal, se eliminan columnas con varianza inferior al umbral configurado (`min_variance: 0.01` en `config/features_config.yaml`). Una columna con varianza < 0.01 significa que al menos el 99% de los registros tienen el mismo valor — esa columna no puede contribuir a distinguir entre positivos y negativos.

### 2.2 Inferencia de features candidatas

La función `infer_candidate_features()` en `src/features/engineering.py` identifica automáticamente qué columnas del `merged.parquet` son candidatas para el modelo, excluyendo:
- Columnas identificadoras (`Documento PMD`, `CODIGO`)
- Columnas que son el target o subflags del target (`target`, `n_flags_relevant`)
- Columnas no numéricas que no pudieron ser codificadas

### 2.3 Score combinado MI + Random Forest

El rankeo principal usa **dos métricas independientes** que se combinan:

#### Mutual Information (MI)
La Información Mutua entre cada feature y el target mide cuánta información sobre el target se puede obtener conociendo el valor de esa feature. A diferencia de la correlación de Pearson, MI captura relaciones no lineales y no asume distribuciones específicas.

```python
mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
```

Valores MI más altos = la feature predice mejor el target.

**Limitación:** MI tiende a sobrevalorar features con muchas categorías (alta cardinalidad) y puede ser ruidosa con pocas observaciones.

#### Feature Importance de Random Forest
Se entrena un Random Forest específicamente para el rankeo (no el modelo final), y se usa la importancia media de disminución de impureza de cada feature como medida de relevancia:

```python
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_split=20,
    min_samples_leaf=10, class_weight="balanced",
)
rf.fit(X, y)
importances = rf.feature_importances_
```

La importancia RF captura interacciones entre features que MI no puede capturar.

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

Los resultados están en `output/v1/reports/pre_post_signal/pre_post_linkage_per_flag.csv`.

### Hallazgos del análisis de señal

**Para `target_d_v2_hosp`:**
- **Max MI:** 0.100 (score_proc_high_severity)
- **Max correlación de Pearson:** 0.232 (score_proc_high_severity)
- **N features con señal real (score combinado > umbral):** 16

**Para `target_d_v2` (sin hospitalización):**
- **Max MI:** 0.031
- **Max correlación de Pearson:** 0.154
- **N features informativas:** 11

La diferencia entre versiones es reveladora: `target_d_v2_hosp` tiene el doble de MI máxima y 50% más de correlación que `target_d_v2`. Esto confirma que añadir `flag_hospitalizacion_no_anticipada` al target **mejora sustancialmente la señal disponible para el modelo** — el target es genuinamente más predecible.

**Top 3 features por señal para `target_d_v2_hosp`:**
1. `score_proc_high_severity` — MI: 0.100, Pearson: 0.232
2. `score_proc_moderate_severity` — MI alto, Pearson moderado
3. `score_proc_medium_severity` — MI alto, Pearson moderado

Los scores de severidad del procedimiento dominan la señal. Esto indica que **la complejidad del procedimiento es el predictor más fuerte del target**, más que las comorbilidades del paciente.

---

## 4. Las 80 features seleccionadas

### 4.1 Distribución por tipo

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

### 4.2 Las 80 features ordenadas por posición (ranking descendente de Combined_Score)

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

### 4.3 Interpretación del ranking

**El tipo de procedimiento y diagnóstico dominan el ranking.** Las primeras 14 features son scores de severidad y tipo de anestesia propuesta — todas derivadas de la complejidad del procedimiento y diagnóstico, no de características del paciente. Esto tiene una implicación importante: el modelo está prediciéndolo principalmente en función de **qué se va a hacer** más que **quién es el paciente**.

**Las comorbilidades aparecen tarde en el ranking.** `Antecedentes cardiovasculares_hta` aparece en la posición 64. `Antecedente endocrinológicos_diabetes` no aparece entre las 80 — está por debajo del umbral, aunque en el análisis de subgrupos (Enfoque B) vimos que los pacientes diabéticos son precisamente los que el modelo predice peor. Esto es coherente: si diabetes tiene baja MI con el target actual, el modelo no aprende a usarla, y por tanto falla en pacientes endocrinológicos.

**Las variables del examen físico tienen señal moderada.** `Edad`, `IMC`, `TA Sistólica`, `Hemoglobina`, `Temperatura`, `Frecuencia Respiratoria` aparecen en el tercio medio del ranking. Son relevantes pero no dominantes.

**Variables temporales tienen señal baja pero real.** `anio`, `mes`, `dia_semana` y `Hora_decimal` superan el umbral mínimo. Hay una pequeña tendencia temporal en los datos (años más recientes, ciertos meses) que el modelo puede explotar, posiblemente relacionada con cambios en protocolos o tipos de pacientes atendidos.

---

## 5. El diagnóstico principal: el modelo predice "complejidad del procedimiento" más que "riesgo del paciente"

El análisis de features revela un patrón preocupante desde el punto de vista de la utilidad clínica del modelo:

**Los predictores más importantes son todos relacionados con la complejidad del procedimiento y el diagnóstico principal** (`score_proc_*`, `score_dx_*`, tipo de anestesia propuesta). Estos factores son informativos para predecir complicaciones, pero son también los más obvios — cualquier médico sabe que un procedimiento de alta complejidad tiene mayor riesgo de complicaciones.

**Las comorbilidades del paciente** — lo que una valoración preanestésica formal realmente evalúa — tienen señal débil en el modelo actual. La diabetes, HTA, EPOC, insuficiencia renal no dominan el modelo.

**Implicación:** El modelo actual puede estar aprendiendo principalmente "procedimiento complejo → complicación probable" en lugar de "paciente de alto riesgo → necesita valoración formal". Esto es útil pero limitado, porque la complejidad del procedimiento es conocida de antemano sin necesidad de un modelo de ML — se puede consultar directamente el tipo de procedimiento.

El verdadero valor añadido de la valoración preanestésica está en identificar a pacientes con factores de riesgo no obvios — el diabético con mal control metabólico, el hipertenso con función renal comprometida, el obeso con apnea del sueño. Estos son los pacientes que una valoración formal cambia el manejo y para los cuales el modelo actual tiene menor señal.

Este diagnóstico es consistente con los hallazgos del Enfoque B (el modelo falla más en pacientes endocrinológicos y cardiovasculares) y del Enfoque C (las comorbilidades crónicas son predictores débiles de muchos flags).

---

## 6. Señal disponible en perspectiva

| Versión target | Max MI | Max Pearson | N features informativas |
|---------------|--------|-------------|------------------------|
| `target_d_v2` | 0.031 | 0.154 | 11 |
| `target_d_v2_hosp` | **0.100** | **0.232** | **16** |
| `target_d_v5` | 0.107 | 0.251 | 16 |

En términos absolutos, incluso la mejor señal (Pearson 0.23) es moderada. Para comparación, una feature con Pearson = 1.0 predice el target perfectamente; con 0.23 ya hay información real pero se necesita combinar muchas features para obtener buena discriminación.

Esto explica por qué todos los modelos se estancan alrededor de AUC 0.75–0.76 independientemente del algoritmo: **el techo de rendimiento está determinado por la señal disponible en los datos, no por el algoritmo**.
