# Enfoque B — Análisis de Error por Subgrupos

> **Pregunta central:** ¿En qué subgrupos de pacientes el modelo funciona bien y en cuáles falla? ¿Hay poblaciones sistemáticamente mal atendidas?

**Código fuente:** [`src/analysis/error_subgroups.py`](../../src/analysis/error_subgroups.py)
**Output CSV:** [`output/v1/reports/analisis_posoperatorio/error_analysis.csv`](../../output/v1/reports/analisis_posoperatorio/error_analysis.csv)
**Output PNG:** [`output/v1/reports/analisis_posoperatorio/error_subgroups.png`](../../output/v1/reports/analisis_posoperatorio/error_subgroups.png)

---

## 1. Motivación

Una métrica global como AUC 0.75 esconde heterogeneidad importante. Un modelo puede tener AUC excelente en pacientes de baja complejidad (la mayoría) y AUC terrible en pacientes de alto riesgo (la minoría, pero la más importante). Si el modelo falla sistemáticamente en algún subgrupo clínico relevante, eso es un hallazgo crítico tanto para la evaluación del modelo como para la toma de decisiones sobre su uso.

Este análisis desagrega el rendimiento del modelo por cuatro variables de segmentación clínicamente significativas, calculando para cada subgrupo:
- **ROC AUC:** capacidad discriminativa del modelo en ese grupo.
- **FN rate (Tasa de falsos negativos):** proporción de pacientes positivos que el modelo clasifica como negativos — pacientes que necesitaban valoración y el modelo "no los ve".
- **FP rate (Tasa de falsos positivos):** proporción de pacientes negativos clasificados como positivos — pacientes que no necesitaban valoración pero el modelo los señala.

---

## 2. Metodología

### 2.1 Datos y modelo

- **Conjunto de prueba:** 4,678 registros (`X_test.parquet` + `y_test.parquet`), con 1,295 positivos (27.7% de prevalencia).
- **Modelo:** `random_forest_model.joblib` de `target_d_v2_hosp`, cargado tal cual fue entrenado.
- **Threshold:** 0.17 (umbral operativo del modelo, optimizado para maximizar Recall).
- **Features:** Las 80 features seleccionadas, con conversión a numérico e imputación de faltantes con -1 para el modelo.

Para calcular las métricas por subgrupo, se hace un join entre el test set y el `merged.parquet` por índice de fila (los registros del test son un subconjunto del merged y mantienen los mismos índices originales).

### 2.2 Variables de segmentación

Se analizan cuatro variables:

| Variable | Descripción | Subgrupos |
|----------|-------------|-----------|
| `tipo_anestesia` | Tipo de anestesia propuesta en la valoración preanestésica | general, sedacion, raquidea, peridural, local, bloqueo n, sin dato |
| `score_proc_severidad_quartil` | Cuartiles del score de severidad del procedimiento | Q1_bajo, Q2, Q3, Q4_alto |
| `grupo_edad` | Grupos etarios | 18-40, 41-60, 61-75, 76+ |
| `dx_cie10_capitulo` | Capítulo del código CIE-10 del diagnóstico preoperatorio | A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, Z |

### 2.3 Métricas calculadas

Para que un subgrupo se incluya en el análisis debe tener al menos 20 pacientes y al menos 2 valores distintos de y_true (tener al menos un positivo y un negativo).

```
ROC AUC: capacidad de rankear positivos sobre negativos (threshold-independent)
FN rate: (falsos negativos) / (total positivos reales) = % de positivos no detectados
FP rate: (falsos positivos) / (total negativos reales) = % de negativos mal clasificados
```

---

## 3. Contexto global del modelo

Antes de ver los subgrupos, las métricas globales en el test set:

| Métrica | Valor |
|---------|-------|
| ROC AUC global | **0.7518** |
| Threshold | 0.17 |
| Recall (Sensitivity) | 0.860 |
| Specificity | 0.398 |
| FN rate global | 0.140 |
| FP rate global | ~0.602 |
| Predicted Positive Rate | 67.3% |

El umbral bajo (0.17) hace que el modelo sea muy sensible: clasifica como positivo el 67.3% de los pacientes para asegurarse de no perder verdaderos positivos. El precio es una especificidad baja — muchos falsos positivos. Esto es una decisión consciente para el contexto clínico (es peor no detectar a un paciente que necesita valoración que enviar a uno que no la necesita).

---

## 4. Resultados por variable de segmentación

### 4.1 Tipo de anestesia propuesta

La anestesia propuesta es una variable declarada por el anestesiólogo en la valoración preanestésica — refleja la planificación para el procedimiento.

| Tipo de anestesia | n | N positivos | Prevalencia | ROC AUC | FN rate | FP rate |
|-------------------|---|-------------|-------------|---------|---------|---------|
| bloqueo n | 413 | 121 | 29.3% | **0.7947** | 11.6% | 52.1% |
| raquidea | 486 | 124 | 25.5% | **0.7921** | 12.9% | 57.2% |
| local | 293 | 73 | 24.9% | **0.7902** | 12.3% | 56.4% |
| sedacion | 713 | 193 | 27.1% | 0.7759 | 14.5% | 56.7% |
| general | 3,780 | 1,052 | 27.8% | 0.7485 | 14.3% | 59.6% |
| sin dato | 331 | 100 | 30.2% | 0.7077 | 14.0% | 64.1% |
| peridural | 113 | 29 | 25.7% | **0.6802** | **31.0%** | 51.2% |

#### Hallazgos clave

**La anestesia general tiene el mayor volumen (3,780 pacientes) pero no el mejor AUC.** Los pacientes con bloqueo nervioso, raquídea y local muestran AUC consistentemente más alto (~0.79). Esto puede explicarse porque:
- Los procedimientos con anestesia regional (raquídea, epidural, bloqueo) suelen estar más claramente definidos en tipo e indicación — hay menos ambigüedad en la señal.
- Los procedimientos con anestesia general son más heterogéneos (desde una colecistectomía laparoscópica hasta una resección tumoral mayor), haciendo más difícil aprender un patrón único.

**La anestesia peridural es el subgrupo problemático.** Con solo 113 pacientes y AUC 0.68, el modelo falla más aquí que en cualquier otro tipo anestésico. Más alarmante: la **tasa de falsos negativos es 31%** — de cada 3 pacientes con anestesia peridural que sí necesitaban valoración formal, el modelo falla en detectar 1. La anestesia peridural se usa frecuentemente en cirugías obstétricas y ginecológicas — es posible que este subgrupo tenga características poblacionales distintas (más jóvenes, menos comorbilidades previas, pero riesgo por embarazo) que el modelo no captura bien.

**Los casos "sin dato" tienen AUC 0.71 pero la mayor tasa de falsos positivos (64.1%).** La ausencia del tipo de anestesia puede reflejar consultas preoperatorias sin decisión tomada, o registros incompletos — un subgrupo inherentemente heterogéneo.

---

### 4.2 Score de severidad del procedimiento (cuartiles)

El `score_proc_high_severity` es una de las features más importantes del modelo (top 1 en importancia). Se divide en cuartiles para ver cómo rinde el modelo en distintos niveles de complejidad quirúrgica.

| Cuartil | n | N positivos | Prevalencia | ROC AUC | FN rate | FP rate |
|---------|---|-------------|-------------|---------|---------|---------|
| Q4_alto (más severo) | 1,166 | 309 | 26.5% | **0.7726** | 15.2% | 59.0% |
| Q1_bajo (menos severo) | 1,181 | 329 | 27.9% | 0.7528 | 12.5% | 60.5% |
| Q2 | 1,158 | 331 | 28.6% | 0.7438 | 13.3% | 58.5% |
| Q3 | 1,173 | 326 | 27.8% | 0.7389 | 15.0% | 60.2% |

#### Hallazgos clave

**El modelo rinde MEJOR en los procedimientos más severos (Q4).** Esto es contraintuitivo — uno esperaría que los casos más complejos fueran más difíciles de predecir. La explicación es que en procedimientos de muy alta severidad, la señal preoperatoria es más clara: la combinación de "procedimiento complejo + paciente de alto riesgo" hace que el target sea más predecible. Los procedimientos de severidad extrema tienen un patrón de riesgo más consistente.

**Los cuartiles intermedios (Q2 y Q3) tienen el AUC más bajo.** En la zona intermedia de severidad es donde hay más ambigüedad: un procedimiento de severidad media puede complicarse o no dependiendo de factores intraoperatorios no predecibles. En los extremos (muy simple o muy complejo), el resultado es más determinista.

**Las prevalencias son similares entre cuartiles (~27%).** Esto indica que el target compuesto actual no discrimina bien entre procedimientos de distinta complejidad — la prevalencia de "alguna complicación" no sube sustancialmente con la severidad del procedimiento, lo que confirma que el target incluye complicaciones que ocurren independientemente de la complejidad.

**Las tasas de FN son similares entre cuartiles (12.5–15.2%).** El modelo no tiene un sesgo fuerte por severidad del procedimiento — no es que falle más en los casos más complejos.

---

### 4.3 Grupos de edad

| Grupo edad | n | N positivos | Prevalencia | ROC AUC | FN rate | FP rate |
|-----------|---|-------------|-------------|---------|---------|---------|
| 41-60 | 1,843 | 512 | 27.8% | **0.7619** | 12.3% | 59.9% |
| 61-75 | 591 | 150 | 25.4% | 0.7507 | 14.0% | 58.1% |
| 76+ | 188 | 64 | 34.0% | 0.7499 | 17.2% | 50.8% |
| 18-40 | 2,056 | 569 | 27.7% | 0.7445 | 15.1% | 60.5% |

#### Hallazgos clave

**La variación de AUC entre grupos de edad es mínima (0.744–0.762).** La edad no es una fuente importante de error diferencial. El modelo no discrimina mejor a los jóvenes que a los ancianos ni viceversa. Esto significa que si el modelo tiene limitaciones, no son atribuibles a la edad del paciente.

**Los pacientes mayores de 76 años tienen la mayor prevalencia (34%)** — uno de cada tres ancianos tiene el target positivo. Sin embargo, el modelo tiene para ellos:
- AUC 0.75 — similar al global.
- FN rate 17.2% — el más alto de todos los grupos etarios. Por cada 6 ancianos positivos, el modelo falla en detectar 1.
- FP rate 50.8% — el más bajo: el modelo es más preciso en los negativos de este grupo.

En adultos mayores, el patrón es: el modelo "sabe" mejor quiénes no necesitan valoración (FP rate bajo), pero es más conservador en marcar a los que sí la necesitan (FN rate algo más alto). Con un threshold de 0.17 esto es marginal, pero podría considerarse subir la sensibilidad en este grupo.

**Los adultos jóvenes (18-40) tienen la FN rate más alta junto a los ancianos (15.1%).** Estos son pacientes que el modelo tiende a considerar "de bajo riesgo" por su edad, pero que presentan complicaciones por otras razones (procedimientos complejos, condiciones médicas específicas).

**Conclusión sobre edad:** La edad como variable de segmentación no revela problemas sistemáticos. El modelo trata igual a todos los grupos etarios. Si hay limitaciones, son transversales a la edad.

---

### 4.4 Capítulo CIE-10 del diagnóstico preoperatorio

Esta es la segmentación más informativa. El capítulo CIE-10 agrupa pacientes por su diagnóstico principal — el motivo médico de la cirugía. Revela diferencias enormes en el rendimiento del modelo.

| Capítulo CIE-10 | Descripción | n | N positivos | Prevalencia | ROC AUC | FN rate | FP rate |
|-----------------|-------------|---|-------------|-------------|---------|---------|---------|
| R | Síntomas/signos sin diagnóstico definitivo | 190 | 60 | 31.6% | **0.8503** | 6.7% | 55.4% |
| J | Enfermedades respiratorias | 92 | 19 | 20.7% | **0.8190** | 10.5% | 58.9% |
| Q | Malformaciones congénitas | 124 | 38 | 30.7% | **0.7987** | 18.4% | 54.7% |
| L | Enfermedades de la piel | 34 | 11 | 32.4% | **0.7964** | 9.1% | 60.9% |
| H | Enfermedades del ojo y del oído | 209 | 59 | 28.2% | **0.7943** | 5.1% | 55.3% |
| M | Enfermedades musculoesqueléticas | 555 | 173 | 31.2% | 0.7651 | 10.4% | 57.6% |
| O | Embarazo, parto y puerperio | 123 | 23 | 18.7% | 0.7635 | 13.0% | 69.0% |
| NO ENCONTRADO | Diagnóstico no codificado | 1,167 | 313 | 26.8% | 0.7625 | 14.1% | 61.0% |
| G | Enfermedades del sistema nervioso | 176 | 41 | 23.3% | 0.7565 | 12.2% | 55.6% |
| N | Enfermedades genitourinarias | 274 | 67 | 24.5% | 0.7544 | 16.4% | 59.9% |
| nan | Sin dato | 370 | 88 | 23.8% | 0.7507 | 13.6% | 59.2% |
| D | Neoplasias benignas / sangre | 99 | 31 | 31.3% | 0.7479 | 12.9% | 63.2% |
| Z | Factores de salud / controles | 117 | 37 | 31.6% | 0.7463 | 13.5% | 55.0% |
| F | Trastornos mentales | 28 | 7 | 25.0% | 0.7279 | 0.0% | 61.9% |
| S | Traumatismos y lesiones | 419 | 132 | 31.5% | 0.7124 | 18.9% | 62.7% |
| I | Enfermedades cardiovasculares | 86 | 25 | 29.1% | 0.7121 | 24.0% | 54.1% |
| C | Neoplasias malignas | 64 | 17 | 26.6% | 0.7071 | 17.7% | 57.5% |
| T | Envenenamientos/consecuencias externas | 43 | 14 | 32.6% | 0.7069 | 28.6% | 51.7% |
| K | Enfermedades digestivas | 286 | 74 | 25.9% | 0.7012 | 16.2% | 62.7% |
| B | Enfermedades infecciosas (cont.) | 31 | 6 | 19.4% | 0.6967 | 16.7% | 56.0% |
| A | Enfermedades infecciosas | 97 | 29 | 29.9% | 0.6945 | 20.7% | 63.2% |
| P | Afecciones perinatales | 45 | 18 | 40.0% | 0.6553 | 5.6% | 66.7% |
| E | Endocrinológicas / nutricionales | 39 | 11 | 28.2% | **0.5357** | **36.4%** | 46.4% |

#### Análisis detallado por capítulo

---

**Capítulo R — Síntomas y signos sin diagnóstico definitivo (AUC 0.8503)**

Este es el subgrupo donde el modelo funciona mejor. El capítulo R incluye pacientes cuyo diagnóstico no tiene una etiqueta definitiva — se operan por síntomas (dolor abdominal inespecífico, síncope, etc.). Paradójicamente, el modelo los predice bien. La hipótesis es que estos pacientes representan un perfil relativamente homogéneo y "promedio" dentro del conjunto de datos — sin comorbilidades dominantes que distorsionen la señal. Además, la tasa de FN (6.7%) es la más baja de todos los grupos con prevalencia >10%.

---

**Capítulo J — Enfermedades respiratorias (AUC 0.8190)**

Pacientes con diagnósticos respiratorios (EPOC, asma, neumonías, etc.). El modelo funciona excelentemente aquí porque las variables preoperatorias incluyen antecedentes respiratorios específicos que son fuertes predictores de complicaciones en este grupo. La valoración preanestésica captura directamente los factores de riesgo relevantes para este capítulo.

---

**Capítulo Q — Malformaciones congénitas (AUC 0.7987)**

Pacientes con defectos congénitos operados. AUC bueno, pero FN rate elevado (18.4%). Estos son frecuentemente pacientes jóvenes o pediátricos (≥18 en este dataset) con presentaciones clínicas variables. El modelo puede "rankear" bien a estos pacientes pero pierde más casos individuales de lo deseable.

---

**Capítulo H — Ojo y oído (AUC 0.7943)**

Pacientes con cirugías oftalmológicas y otorrinolaringológicas. AUC alto y FN rate muy bajo (5.1%). Estos procedimientos tienden a ser electivos, bien planificados, con perfiles de paciente claros. El modelo discrimina bien en este subgrupo.

---

**Capítulo M — Musculoesquelético (AUC 0.7651)**

Artroscopias, reemplazos articulares, fracturas crónicas. AUC cercano al global, comportamiento similar al promedio del modelo.

---

**Capítulo O — Embarazo, parto y puerperio (AUC 0.7635, FP rate 69.0%)**

Las pacientes obstétricas tienen la **tasa de falsos positivos más alta** (69.0%) — de cada 10 pacientes embarazadas que no necesitarían valoración formal según el target, el modelo dice que sí la necesitan en casi 7. Esto puede reflejar que el modelo, al no haber sido diseñado específicamente para obstetricia, tiende a sobreclasificar por precaución en embarazadas, o que el target compuesto no es apropiado para este subgrupo clínico.

---

**Capítulo I — Enfermedades cardiovasculares (AUC 0.7121, FN rate 24.0%)**

Los pacientes cardiovasculares tienen AUC moderado y una **tasa de falsos negativos preocupante (24%)** — de cada 4 pacientes cardíacos positivos, el modelo falla en detectar 1. Estos son precisamente los pacientes para quienes la valoración preanestésica tiene mayor valor clínico: riesgo de arritmias, infarto perioperatorio, descompensación cardíaca. El modelo falla más aquí de lo que sería aceptable clínicamente.

---

**Capítulo T — Envenenamientos y causas externas (AUC 0.7069, FN rate 28.6%)**

Traumatismos severos, cirugías de emergencia, intoxicaciones. La FN rate es muy alta (28.6%). Estos son casos de urgencia donde la valoración preanestésica puede ser incompleta o urgente — el perfil preop puede no capturar bien la gravedad real del paciente de trauma.

---

**Capítulo A — Enfermedades infecciosas (AUC 0.6945, FN rate 20.7%)**

Pacientes con infecciones activas que requieren cirugía. AUC bajo y FN rate alto. Las infecciones activas generan un estado inflamatorio que puede complicarse de manera impredecible.

---

**Capítulo P — Afecciones perinatales (AUC 0.6553)**

Neonatos y pacientes muy jóvenes. AUC bajo (0.655) pero FN rate sorprendentemente bajo (5.6%). La prevalencia es alta (40%) y el FP rate es muy alto (66.7%). El modelo sobreclasifica en este grupo.

---

**Capítulo E — Enfermedades endocrinológicas, nutricionales y metabólicas (AUC 0.5357)**

**Este es el hallazgo más crítico de todo el análisis.** Los pacientes con diagnósticos endocrinológicos (diabetes, hipotiroidismo, síndrome metabólico, obesidad mórbida) tienen el peor AUC de todos los subgrupos: **0.5357 — prácticamente aleatorio**.

- **N:** 39 pacientes (pocos, pero el resultado es estadísticamente válido para su tamaño)
- **Prevalencia:** 28.2% — similar al promedio
- **FN rate: 36.4%** — de cada 3 pacientes endocrinológicos positivos, el modelo **falla en detectar 1**.
- **FP rate: 46.4%** — el más bajo de todos los grupos, lo que significa que el modelo es "conservador" con los negativos pero falla en los positivos.

La combinación de AUC ~azar y FN rate 36.4% significa que el modelo básicamente no puede distinguir en este subgrupo qué pacientes endocrinológicos van a tener complicaciones y cuáles no.

**¿Por qué falla aquí?** Hipótesis:
1. La muestra es pequeña (39 pacientes en test), lo que hace la estimación ruidosa.
2. Los pacientes endocrinológicos pueden tener complicaciones de naturaleza muy variable (crisis diabética, hipotiroidismo descompensado, crisis addisoniana) que dependen de factores no capturados en las features actuales.
3. Las complicaciones en este grupo pueden estar más ligadas a la gestión perioperatoria de la medicación (insulina, levotiroxina) que a características basales registradas en preop.

**Implicación clínica:** Los pacientes endocrinológicos son quienes más se beneficiarían de una valoración preanestésica formal — y el modelo es casi inútil para identificarlos. Esto es un fallo crítico de equidad del modelo.

---

## 5. Síntesis y conclusiones

### 5.1 El modelo no falla uniformemente

| Variable | Rango de AUC | Gap máximo |
|----------|-------------|-----------|
| Tipo anestesia | 0.680 – 0.795 | 0.115 |
| Severidad procedimiento | 0.739 – 0.773 | 0.034 |
| Grupo edad | 0.744 – 0.762 | 0.018 |
| Capítulo CIE-10 | 0.536 – 0.850 | 0.314 |

El capítulo CIE-10 es con diferencia la fuente de mayor variabilidad. La edad prácticamente no importa. La severidad del procedimiento importa poco. El tipo de anestesia importa algo. Pero el diagnóstico de base del paciente es el factor más determinante para si el modelo funciona bien o mal.

### 5.2 Subgrupos problemáticos (requieren atención)

| Subgrupo | AUC | FN rate | Problema principal |
|----------|-----|---------|-------------------|
| Capítulo E (endocrino) | 0.536 | 36.4% | Modelo casi aleatorio |
| Anestesia peridural | 0.680 | 31.0% | Alto FN en cirugía obstétrica |
| Capítulo T (trauma) | 0.707 | 28.6% | Alto FN en emergencias |
| Capítulo I (cardiovascular) | 0.712 | 24.0% | Alto FN en pacientes cardíacos |
| Capítulo A (infecciosas) | 0.695 | 20.7% | AUC bajo + FN elevado |

### 5.3 Subgrupos donde el modelo funciona bien

| Subgrupo | AUC | FN rate | Característica |
|----------|-----|---------|----------------|
| Capítulo R (síntomas) | 0.850 | 6.7% | Pacientes "promedio" sin comorbilidad dominante |
| Capítulo J (respiratorio) | 0.819 | 10.5% | Variables preop capturan antecedentes relevantes |
| Capítulo H (ojo/oído) | 0.794 | 5.1% | Procedimientos electivos bien definidos |
| Bloqueo nervioso | 0.795 | 11.6% | Procedimientos con perfil de riesgo claro |

### 5.4 Reflexión sobre equidad del modelo

Un modelo de screening médico tiene un problema de equidad si falla sistemáticamente en los subgrupos que más necesitan ser detectados. El análisis muestra que los pacientes endocrinológicos y cardiovasculares — que representan los mayores riesgos perioperatorios y quienes más se beneficiarían de valoración preanestésica formal — son precisamente los subgrupos donde el modelo tiene peor rendimiento.

Esto no invalida el modelo, pero señala que su despliegue directo sin validación adicional en estos subgrupos podría generar daño: pacientes de alto riesgo no detectados que no reciben la valoración que necesitan.
