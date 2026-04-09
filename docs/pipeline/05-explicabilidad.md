# Etapa 5 — Explicabilidad del Modelo

**Código fuente:**
- [`src/evaluation/explainability.py`](../../src/evaluation/explainability.py) — Cálculo de importancias globales y locales
- [`src/reports/pre_post_analysis.py`](../../src/reports/pre_post_analysis.py) — Análisis de señal preop→posop

**Outputs:**
- `output/v1/reports/explainability/target_d_v2_hosp/explainability_global_{modelo}.csv` — Importancias globales por feature
- `output/v1/reports/explainability/target_d_v2_hosp/explainability_cases_{modelo}.csv` — Importancias locales por caso

---

## 1. ¿Por qué es necesaria la explicabilidad?

En un contexto médico, un modelo de ML no puede ser una caja negra. Los médicos y administradores del hospital necesitan saber:

1. **¿Por qué el modelo dice que este paciente necesita valoración?** — Explicación local (caso por caso).
2. **¿Qué factores son más importantes para el modelo en general?** — Explicación global.
3. **¿El modelo está usando las variables "correctas" desde el punto de vista clínico?** — Validación de sensatez.
4. **¿Hay sesgos en el modelo?** — Por ejemplo, ¿está discriminando por sexo o por grupo étnico?

La explicabilidad también es necesaria para identificar problemas del modelo: si el factor más importante es una variable proxy de algo que el modelo no debería usar, eso indica fuga de datos u otros problemas.

---

## 2. Importancia global de features — Permutation Importance

La técnica principal de explicabilidad global es la **Importancia por Permutación** (Permutation Importance).

### ¿Cómo funciona?
1. Se evalúa el modelo en el test set y se registra el AUC base.
2. Se baraja aleatoriamente **una feature a la vez** (rompiendo su relación con el target).
3. Se vuelve a evaluar el modelo con esa feature permutada.
4. La caída en AUC al permutar la feature = su importancia de permutación.

Una feature con importancia 0.04 significa que permutarla hace caer el AUC en 0.04 — es una feature que el modelo usa activamente y que contribuye 0.04 puntos al AUC.

### ¿Por qué permutation importance en lugar de feature importances de árbol?

La importancia nativa de los árboles de decisión (MDI, Mean Decrease in Impurity) tiene sesgos conocidos: sobrestima la importancia de features numéricas continuas y de alta cardinalidad. La permutation importance no tiene este sesgo — mide el impacto real en la métrica de evaluación.

---

## 3. Resultados de explicabilidad global — Random Forest

Las 20 features con mayor importancia de permutación para el Random Forest en `target_d_v2_hosp`:

| Posición | Feature | Importancia de permutación |
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

---

## 4. Análisis profundo de las variables más importantes

### 4.1 `Tipo de anestesia propuesta_sedacion` (importancia: 0.044) — El predictor más importante

Esta es la variable con mayor importancia de permutación en el modelo. Que "anestesia propuesta = sedación" sea el predictor más importante llama la atención — ¿qué información captura?

**Interpretación:** Los procedimientos realizados con sedación son típicamente procedimientos diagnósticos o intervencionistas menores (colonoscopia, endoscopia, cateterismos, biopsias bajo sedación). En este contexto:
- La sedación se asocia con **procedimientos de baja complejidad** → baja probabilidad de complicaciones graves.
- Si el paciente va a sedación, el modelo aprende que el riesgo de hospitalización no anticipada, UCI, o complicaciones mayores es menor.
- Permutando esta variable (mezclando aleatoriamente quién va a sedación y quién no) destruye esta señal — el modelo pierde 0.04 en AUC.

**Implicación clínica:** El modelo está usando el tipo de anestesia propuesta como proxy de la complejidad del procedimiento. Esto tiene sentido clínico: el tipo de anestesia es determinado en gran parte por el tipo de procedimiento.

**Posible problema:** Si el tipo de anestesia es determinado *durante* la valoración preanestésica (en la misma consulta cuyos datos se usan como features), podría haber un elemento de circularidad. El modelo puede estar aprendiendo "el anestesiólogo ya decidió anestesia raquídea → el procedimiento es de cierto tipo → el riesgo es determinado". En ese caso, la feature no añade información adicional a la que el anestesiólogo ya tiene.

---

### 4.2 `Tipo de anestesia propuesta_raquidea` (importancia: 0.034) — Segundo predictor

La anestesia raquídea (espinal) se usa para procedimientos en miembros inferiores y abdomen bajo: cesáreas, cirugías ortopédicas de cadera/rodilla, herniorrafias, etc. Tiene características de riesgo específicas:
- Es una técnica regional que evita los riesgos de la intubación endotraqueal.
- Los pacientes que reciben raquídea tienen perfiles de riesgo distintos a los de anestesia general.
- Las complicaciones de la raquídea son diferentes a las de la anestesia general.

Al igual que con la sedación, el modelo usa la raquídea como proxy del tipo de procedimiento y su perfil de riesgo asociado.

---

### 4.3 `Tipo de anestesia propuesta_peridural` (importancia: 0.017)

Similar a la raquídea pero para procedimientos más prolongados y con infusión continua (trabajo de parto, cirugías extensas). El impacto en AUC es menor que raquídea, pero aún significativo.

---

### 4.4 `Edad` (importancia: 0.0069) — Cuarta feature más importante

La edad es la variable continua más importante. Es el predictor clínico más intuitivo del riesgo perioperatorio: a mayor edad, mayor prevalencia de comorbilidades, menor reserva fisiológica y mayor riesgo de complicaciones.

Sin embargo, el análisis de subgrupos (Enfoque B) mostró que el AUC varía poco entre grupos de edad (0.744–0.762). Esto significa que la edad aporta información al modelo pero no es la fuente principal de sus errores ni de sus aciertos.

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

Al analizar el conjunto de las 20 features más importantes, emerge un patrón claro:

**El modelo aprende principalmente:**
1. **El tipo de anestesia propuesta** (features 1, 2, 3, 10) — proxy de complejidad del procedimiento.
2. **La severidad del procedimiento** (features 5, 8, 11, 12, 13) — complejidad directa.
3. **La severidad del diagnóstico** (features 9, 14, 18) — gravedad de la condición base.
4. **El tipo de diagnóstico** (features 7, 15, 16, 20) — capítulo CIE-10.
5. **Variables demográficas y de examen** (features 4, 6, 19) — Edad, Hemoglobina, Sexo.

**Notablemente ausentes de las top 20:**
- HTA (`Antecedentes cardiovasculares_hta`) — aparece en posición 64 del ranking de selección
- Diabetes (`Antecedente endocrinológicos_diabetes`) — debajo del umbral de selección
- EPOC, insuficiencia renal, obesidad — no en top 20
- Mallampati — posición 58 en el ranking

**Conclusión:** El modelo predice principalmente en función de QUÉ se va a hacer (procedimiento y diagnóstico) más que en función del estado clínico del paciente. Las comorbilidades que una valoración preanestésica evalúa específicamente (HTA, diabetes, función cardíaca, respiratoria) no son los predictores dominantes.

Esto es consistente con el análisis de subgrupos del Enfoque B, donde los pacientes endocrinológicos (AUC=0.54) y cardiovasculares (FN rate=24%) son los subgrupos con peor rendimiento.

---

## 6. Explicabilidad local — Análisis por casos

Además de la importancia global, se calcula la importancia local para casos individuales. El archivo `explainability_cases_{modelo}.csv` contiene, para cada paciente del test set, qué features contribuyeron más a que el modelo le asignara esa probabilidad.

Esta información es útil para:
- Explicar a un médico por qué el modelo marcó a un paciente específico como de alto riesgo.
- Identificar si la predicción tiene una justificación clínica coherente o si parece arbitraria.
- Detectar casos donde el modelo se apoya en features inesperadas (potencial señal de problema).

---

## 7. Implicaciones para el despliegue clínico

### 7.1 Transparencia necesaria

Antes de desplegar el modelo en producción, se debe poder responder a preguntas como:
- "¿Por qué este paciente fue marcado?" → Requiere explicación local por caso.
- "¿El modelo no discrimina por sexo o raza?" → Requiere análisis de equidad por subgrupo.
- "¿Es coherente con el juicio clínico?" → Requiere validación con anestesiólogos.

### 7.2 Limitaciones identificadas

1. **El modelo predice complejidad del procedimiento más que riesgo del paciente.** Un modelo que simplemente mira el tipo de procedimiento y anestesia propuesta podría aproximar el 80% del rendimiento del modelo de ML. El valor añadido real está en la integración de comorbilidades, examen físico y laboratorios — que actualmente tienen señal débil.

2. **Los pacientes de alto riesgo médico son los menos detectados.** Pacientes endocrinológicos, cardiovasculares y de trauma tienen las peores métricas, precisamente los subgrupos que más se benefician de valoración formal.

3. **La hemoglobina puede estar siendo usada como indicador de "dato ausente".** Si el modelo aprende "hemoglobina baja = dato ausente = perfil específico de paciente", la feature puede ser un artefacto del proceso de limpieza más que información clínica real.

### 7.3 Recomendaciones para la siguiente iteración

Basado en el análisis de explicabilidad:

1. **Investigar y corregir el parseo de hemoglobina** — el valor medio de 4.14 g/dl es incoherente con valores clínicos normales.

2. **Enriquecer con features de comorbilidades más detalladas** — en lugar de flags binarios ("tiene diabetes sí/no"), incluir métricas de control (HbA1c para diabetes, creatinina para función renal, FEVI para función cardíaca).

3. **Redefinir el target** hacia eventos donde las comorbilidades del paciente son predictores más fuertes — `flag_hospitalizacion_no_anticipada`, `flag_glucometria_anormal`, `flag_interconsultas` — como sugiere el Enfoque C.

4. **Considerar modelos separados por tipo de procedimiento** — un modelo para procedimientos de alta complejidad y otro para procedimientos menores, permitiendo que cada uno aprenda las señales específicas de su subpoblación.
