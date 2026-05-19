# Etapa 5 — Explicabilidad del Modelo (importancias)

**Código fuente:**
- [`src/evaluation/explainability.py`](../../src/evaluation/explainability.py) — Cálculo de importancias globales y locales
- [`src/reports/pre_post_analysis.py`](../../src/reports/pre_post_analysis.py) — Análisis de señal preop→posop
- [`src/reports/shap_plots.py`](../../src/reports/shap_plots.py) — Generación de SHAP values y plots (beeswarm)

**Outputs (pipeline v2):**
- `output/v2/reports/explainability/{target}/explainability_global_{modelo}.csv` — Importancias globales por feature (nativas + permutation fallback)
- `output/v2/reports/explainability/{target}/explainability_cases_{modelo}.csv` — Importancias locales por caso, etiquetadas como TP/FN/FP/TN
- `output/v2/reports/shap/{target}/shap_values_{modelo}.npy` — Matriz SHAP del test set
- `output/v2/reports/shap/{target}/shap_beeswarm_{modelo}.png` — Plot beeswarm de contribuciones SHAP por feature
- `output/v2/reports/shap/{target}/fn_waterfall/` — Plots waterfall por caso falso negativo (generados por `src/reports/shap_plots.plot_shap_waterfall_fn()`)

> **Nota:** Este documento cubre la importancia global y local con técnicas de **permutación e importancia nativa**. El análisis SHAP por **grupo clínico** (TP vs. FN vs. FP vs. TN) tiene su propio documento — ver [07-shap-grupos.md](07-shap-grupos.md).

---

## 1. ¿Por qué es necesaria la explicabilidad?

En un entorno médico, un modelo de ML no puede operar como una caja negra. Los clínicos y administradores del hospital requieren comprender qué factores determinan que un paciente sea marcado como de alto riesgo, qué variables son las más influyentes a nivel global y si el modelo emplea las variables clínicamente correctas. Además, el análisis de explicabilidad es la principal herramienta para detectar problemas de diseño: si el predictor más importante resulta ser una variable proxy de información no disponible en el momento de la valoración, eso puede señalar fuga de datos u otras inconsistencias metodológicas.

---

## 2. Importancia global de features — Importancia nativa (técnica primaria) y por permutación (fallback)

El pipeline calcula importancia global con **dos técnicas**:

1. **Importancia nativa del modelo** (técnica **prioritaria**): `feature_importances_` para modelos de árbol (RF, XGBoost, LightGBM, HGB, ExtraTrees — MDI, Mean Decrease in Impurity) y `coef_` para modelos lineales. El archivo CSV registra en la columna `Source` el valor `native_feature_importance` cuando se usa esta vía.

2. **Importancia por permutación** (**fallback** para modelos sin esos atributos: Stacking, Voting, MLP).

### ¿Cómo funciona?

Se evalúa el modelo en el test set y se registra el AUC base. A continuación, se permuta aleatoriamente una feature a la vez —rompiendo su relación con el target— y se vuelve a evaluar el modelo. La caída en AUC al permutar una feature representa su importancia de permutación: una importancia de 0.04 indica que esa feature contribuye 0.04 puntos al AUC global y que el modelo la utiliza activamente.

La importancia nativa de los árboles (MDI, Mean Decrease in Impurity) tiene sesgos conocidos, en particular la sobrestimación de features numéricas continuas y de alta cardinalidad. La importancia por permutación no presenta ese sesgo porque mide el impacto directo sobre la métrica de evaluación.

---

## 3. Resultados de explicabilidad global

### 3.1 Random Forest sobre `target_d_v2_hosp` (importancia nativa MDI — `feature_importances_`)

Las 20 features con mayor importancia nativa MDI para el Random Forest en `target_d_v2_hosp` (de [`output/v2/reports/explainability/target_d_v2_hosp/explainability_global_random_forest.csv`](../../output/v2/reports/explainability/target_d_v2_hosp/explainability_global_random_forest.csv)):

| Posición | Feature | Importancia MDI |
|----------|---------|---------------------------|
| 1 | `Tipo de anestesia propuesta_sedacion` | **0.04360** |
| 2 | `Tipo de anestesia propuesta_raquidea` | **0.03373** |
| 3 | `Tipo de anestesia propuesta_peridural` | 0.01738 |
| 4 | `Edad` | 0.00694 |
| 5 | `score_proc_medium_severity` | 0.00681 |
| 6 | `Examen_Hemoglobina(g/dl)` | 0.00516 |
| 7 | `Dx Preoperatorio Code_S` | 0.00482 |
| 8 | `score_proc_low_severity` | 0.00465 |
| 9 | `score_dx_critical` | 0.00439 |
| 10 | `Tipo de anestesia propuesta_local` | 0.00411 |
| 11 | `score_proc_critical` | 0.00392 |
| 12 | `score_proc_moderate_severity` | 0.00364 |
| 13 | `score_proc_high_severity` | 0.00363 |
| 14 | `score_dx_low_severity` | 0.00281 |
| 15 | `Dx Preoperatorio Code_A` | 0.00270 |
| 16 | `Dx Preoperatorio Code_Z` | 0.00252 |
| 17 | `Antecedente hematológicos _negativo` | 0.00249 |
| 18 | `score_dx_high_severity` | 0.00181 |
| 19 | `Sexo_encoded` | 0.00178 |
| 20 | `Dx Preoperatorio Code_H` | 0.00177 |

### 3.2 XGBoost sobre `target_f_predictibilidad_maxima` (importancia nativa)

Las 20 features con mayor importancia nativa para el XGBoost en `target_f_predictibilidad_maxima` (de [`output/v2/reports/explainability/target_f_predictibilidad_maxima/explainability_global_xgboost.csv`](../../output/v2/reports/explainability/target_f_predictibilidad_maxima/explainability_global_xgboost.csv)). Estas son las importancias `feature_importances_` en escala normalizada (la suma total sobre **todas** las features del modelo es 1.0; la tabla muestra solo el top-20):

| Posición | Feature | Importancia |
|----------|---------|-------------|
| 1 | `Tipo de anestesia propuesta_raquidea` | **0.1217** |
| 2 | `Tipo de anestesia propuesta_sedacion` | 0.0941 |
| 3 | `Tipo de anestesia propuesta_peridural` | 0.0938 |
| 4 | `Dx Preoperatorio Code_O` (embarazo/parto) | 0.0528 |
| 5 | `Dx Preoperatorio Code_A` (infecciosas) | 0.0451 |
| 6 | `Tipo de anestesia propuesta_general` | 0.0396 |
| 7 | `Dx Preoperatorio Code_Z` (factores de salud) | 0.0337 |
| 8 | `Tipo de anestesia propuesta_sin dato` | 0.0308 |
| 9 | `Dx Preoperatorio Code_S` (traumatismos) | 0.0296 |
| 10 | `Antecedente renales_litiasis` | 0.0290 |
| 11 | `Tipo de anestesia propuesta_local` | 0.0271 |
| 12 | `Antecedente renales_negativo` | 0.0269 |
| 13 | `Antecedente hematológicos_negativo` | 0.0146 |
| 14 | `Procedimiento propuesto Code_A` | 0.0140 |
| 15 | `Grupo Sanguíneo_Sin Dato` | 0.0137 |
| 16 | `Sexo_encoded` | 0.0135 |
| 17 | `Dx Preoperatorio Code_H` (ojo/oído) | 0.0119 |
| 18 | `Dx Preoperatorio Code_K` (digestivo) | 0.0114 |
| 19 | `Examen_Hemoglobina(g/dl)` | 0.0108 |
| 20 | `score_dx_low_severity` | 0.0098 |

En `target_f`, los códigos CIE-10 del diagnóstico ganan peso (4 de las top-10), lo que refleja que el modelo aprende asociaciones específicas entre ciertos diagnósticos y hospitalización o ingreso a UCI más allá del score de severidad agregado. `Antecedente renales_litiasis` asciende a la posición 10 — una patología específica que en `target_d_v2_hosp` quedaba diluida dentro del conjunto. Los scores de severidad agregados retroceden en el ranking de `target_f`: el modelo se apoya más en el tipo de procedimiento y diagnóstico (códigos) que en sus categorías de severidad.

---

## 4. Análisis profundo de las variables más importantes

### 4.1 `Tipo de anestesia propuesta_sedacion` (importancia: 0.044) — El predictor más importante

Esta es la variable con mayor importancia nativa MDI en el modelo Random Forest. Los procedimientos bajo sedación son típicamente procedimientos diagnósticos o intervencionistas menores (colonoscopia, endoscopia, cateterismos, biopsias), de modo que la sedación funciona como proxy de baja complejidad y, por extensión, de menor probabilidad de hospitalización no anticipada, ingreso a UCI o complicaciones mayores. Permutar aleatoriamente esta variable destruye esa señal, haciendo caer el AUC en 0.04.

El tipo de anestesia propuesta actúa, en última instancia, como un proxy de la complejidad del procedimiento quirúrgico, lo cual es clínicamente coherente. Una limitación a considerar es que si el tipo de anestesia se determina durante la misma consulta preanestésica que genera los datos de entrada al modelo, puede existir un elemento de circularidad: el modelo estaría aprendiendo una decisión ya tomada por el anestesiólogo, más que información adicional sobre el riesgo del paciente.

---

### 4.2 `Tipo de anestesia propuesta_raquidea` (importancia: 0.034) — Segundo predictor

La anestesia raquídea se utiliza para procedimientos en miembros inferiores y abdomen bajo: cesáreas, cirugías ortopédicas de cadera y rodilla, herniorrafias, entre otros. Es una técnica regional que evita los riesgos propios de la intubación endotraqueal, y los pacientes que la reciben tienen perfiles de riesgo clínicamente distintos a los de la anestesia general. Al igual que con la sedación, el modelo emplea la anestesia raquídea como proxy del tipo de procedimiento y del perfil de riesgo perioperatorio asociado.

---

### 4.3 `Tipo de anestesia propuesta_peridural` (importancia: 0.017)

Similar a la raquídea pero para procedimientos más prolongados y con infusión continua (trabajo de parto, cirugías extensas). El impacto en AUC es menor que raquídea, pero aún significativo.

---

### 4.4 `Edad` (importancia: 0.0069) — Cuarta feature más importante

La edad es la variable continua más influyente y el predictor clínico más intuitivo del riesgo perioperatorio: a mayor edad, mayor prevalencia de comorbilidades, menor reserva fisiológica y mayor riesgo de complicaciones. Sin embargo, el análisis de subgrupos (Enfoque B) muestra que el AUC varía poco entre grupos etarios (0.744–0.762), lo que indica que la edad aporta información al modelo pero no constituye la fuente principal de sus aciertos ni de sus errores.

---

### 4.5 `score_proc_medium_severity` (importancia: 0.0068)

Los scores de severidad del procedimiento aparecen en posiciones 5, 8, 11, 12 y 13 del ranking. Su importancia individual es relativamente modesta (~0.003–0.007), pero **en conjunto los 5 scores de severidad de procedimiento suman ~0.025** — comparable a la importancia de la sedación o raquídea individualmente.

Esto confirma el diagnóstico del documento de selección de features: la complejidad del procedimiento es el predictor agregado más importante, aunque ninguna categoría individual de severidad domine.

---

### 4.6 `Examen_Hemoglobina(g/dl)` (importancia: 0.0052)

La hemoglobina preoperatoria es un predictor clínicamente válido: la anemia preoperatoria se asocia a mayor riesgo de transfusión, mayor riesgo de complicaciones cardiovasculares y peores desenlaces quirúrgicos. Que aparezca en posición 6 indica que el modelo la usa activamente.

Sin embargo, hay un detalle técnico: la media de hemoglobina en el dataset limpio es **4.14 g/dl** — un valor extremadamente bajo (la hemoglobina normal es 12–17 g/dl). Esto sugiere que hay un problema de parseo o imputación en esta variable para muchos registros. Si la imputación usa un valor centinela (-1 convertido a un valor fuera de rango), la hemoglobina podría estar siendo usada como indicador de "dato de laboratorio ausente" más que como valor clínico real.

---

### 4.7 `Dx Preoperatorio Code_S` — Diagnóstico capítulo S (Traumatismos)

El capítulo S del CIE-10 incluye traumatismos, fracturas y lesiones. Estos pacientes tienen perfiles de riesgo específicos: pueden ser de urgencia, con trauma asociado y múltiples lesiones. Su presencia entre las features importantes sugiere que el modelo ha aprendido un perfil de riesgo diferencial para pacientes traumatizados.

---

## 5. Patrón global de la explicabilidad: ¿qué está aprendiendo el modelo?

Del análisis conjunto de las 20 features más importantes emerge un patrón consistente. El modelo aprende principalmente el tipo de anestesia propuesta (proxy de complejidad del procedimiento), la severidad del procedimiento, la severidad y tipo del diagnóstico (por capítulo CIE-10) y, en menor medida, variables demográficas y de examen como Edad, Hemoglobina y Sexo.

Están notablemente ausentes de las primeras posiciones las comorbilidades que una valoración preanestésica formal evalúa de forma específica: HTA aparece en la posición 64 del ranking de selección; diabetes queda por debajo del umbral de selección; EPOC, insuficiencia renal y obesidad no figuran en el top 20; y la escala de Mallampati ocupa la posición 58. El modelo predice fundamentalmente en función de qué procedimiento se va a realizar, más que del estado clínico del paciente. Esto es coherente con los hallazgos del Enfoque B, donde los pacientes endocrinológicos (AUC=0.54) y cardiovasculares (tasa de FN del 24%) presentan el peor rendimiento — son precisamente los subgrupos donde las comorbilidades deberían tener mayor relevancia predictiva pero tienen señal débil en el modelo actual.

---

## 6. Explicabilidad local — Análisis por casos

Además de la importancia global, se calcula la importancia local para casos individuales. El archivo `explainability_cases_{modelo}.csv` contiene, para cada paciente seleccionado del test set, qué features contribuyeron más a que el modelo le asignara esa probabilidad.

**Cómo se seleccionan los casos:** Por defecto se eligen hasta 10 casos por grupo clínico (TP, FN, FP, TN) usando la columna `case_type`. Los archivos incluyen:
- `case_index`: índice original del DataFrame en el test set.
- `case_type`: clasificación clínica (TP/FN/FP/TN).
- Una columna por feature con la contribución SHAP a la probabilidad final.

**Visualizaciones derivadas:**
- `output/v2/reports/shap/{target}/shap_beeswarm_{modelo}.png` — Beeswarm SHAP estándar para el modelo: cada punto es un paciente, eje X es la contribución SHAP, color por valor de la feature.
- `output/v2/reports/shap/{target}/fn_waterfall/` — Waterfall por paciente FN, mostrando la "ruta" de decisión del modelo desde el valor base hasta la probabilidad final.

Esta información permite explicar a un clínico por qué el modelo marcó a un paciente específico como de alto riesgo, verificar si la predicción tiene una justificación clínicamente coherente y detectar casos en los que el modelo se apoya en features inesperadas — posible indicador de problemas metodológicos. Para un análisis **agregado** por grupo clínico que identifique qué distingue sistemáticamente a los falsos negativos de los verdaderos positivos, ver [07-shap-grupos.md](07-shap-grupos.md).

---

## 7. Implicaciones para el despliegue clínico

### 7.1 Requisitos de transparencia

Antes de desplegar el modelo en producción, el equipo clínico debe poder responder con fundamento a las siguientes preguntas: por qué el modelo marcó a un paciente específico (requiere explicación local por caso), si el modelo discrimina por sexo u otras características protegidas (requiere análisis de equidad por subgrupo) y si sus predicciones son coherentes con el juicio clínico (requiere validación con anestesiólogos experimentados).

### 7.2 Limitaciones identificadas

El análisis de explicabilidad expone tres limitaciones relevantes para el despliegue. Primero, el modelo predice complejidad del procedimiento más que riesgo del paciente: un modelo que simplemente observe el tipo de procedimiento y la anestesia propuesta podría aproximar el 80% del rendimiento del modelo de ML. El valor diferencial reside en la integración de comorbilidades, examen físico y laboratorios, que en la versión actual tienen señal débil. Segundo, los pacientes de mayor riesgo médico — endocrinológicos, cardiovasculares y de trauma — son precisamente los que el modelo detecta con menor fiabilidad, aunque son también los que más se beneficiarían de una valoración formal. Tercero, la hemoglobina puede estar siendo utilizada como indicador de dato ausente más que como valor clínico real: su media de 4.14 g/dl en el dataset limpio es biológicamente incompatible con los rangos normales (12–17 g/dl), lo que sugiere un problema de parseo.

### 7.3 Recomendaciones para la siguiente iteración

El análisis de explicabilidad fundamenta cuatro recomendaciones concretas para iteraciones futuras. Primero, investigar y corregir el parseo de hemoglobina, cuyo valor medio de 4.14 g/dl es incoherente con valores clínicos normales. Segundo, enriquecer las features de comorbilidades con indicadores cuantitativos de control en lugar de flags binarios: HbA1c para diabetes, creatinina para función renal y FEVI para función cardíaca, entre otros. Tercero, redefinir el target hacia eventos donde las comorbilidades del paciente son predictores más fuertes — `flag_hospitalizacion_no_anticipada`, `flag_glucometria_anormal`, `flag_interconsultas` —, como sugiere el Enfoque C. Cuarto, considerar el entrenamiento de modelos separados por tipo de procedimiento, permitiendo que cada submodelo aprenda las señales específicas de riesgo de su subpoblación.
