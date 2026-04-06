# Enfoque C — Predictibilidad de Flags Posoperatorios

> **Pregunta central:** ¿Qué complicaciones o eventos posoperatorios son predecibles desde las variables de la valoración preanestésica?

**Código fuente:** [`src/analysis/flag_predictability.py`](../../src/analysis/flag_predictability.py)
**Output CSV:** [`output/v1/reports/analisis_posoperatorio/flag_predictability.csv`](../../output/v1/reports/analisis_posoperatorio/flag_predictability.csv)
**Output PNG:** [`output/v1/reports/analisis_posoperatorio/flag_predictability.png`](../../output/v1/reports/analisis_posoperatorio/flag_predictability.png)

---

## 1. Motivación

El modelo principal del proyecto predice un **target compuesto**: una variable binaria que vale 1 si el paciente presentó *alguna* de una lista de complicaciones o eventos posoperatorios. El problema con un target compuesto es que puede mezclar eventos de naturaleza muy distinta:

- Complicaciones que dependen de características del paciente conocidas antes de la cirugía (predecibles desde preop).
- Complicaciones que ocurren por razones intraoperatorias — decisiones del cirujano, respuesta inesperada del paciente — que nadie puede anticipar desde los datos de la valoración preanestésica.

Si el target mezcla eventos predecibles con impredecibles, el modelo aprende una señal promediada y su capacidad predictiva queda artificialmente limitada. Este análisis cuantifica exactamente eso: para cada flag posoperatorio por separado, ¿qué tan bien pueden predecirlo las variables preoperatorias?

---

## 2. Metodología

### 2.1 Datos de entrada

Se parte de dos fuentes que se unen por paciente:

- **`merged.parquet`** (`target_d_v2_hosp`): 23,387 registros con variables preoperatorias (valoración preanestésica). Filtrado a mayores de 18 años. Contiene 80 features seleccionadas más el target.
- **`posop_raw.parquet`**: 29,865 registros posoperatorios con 57 flags clínicos binarios. Se une al merged mediante la clave `Documento PMD` ↔ `Documento PMD (valoración preanestésica)`, resultando en **23,387 pacientes con datos completos de ambas fuentes**.

### 2.2 Selección de flags a analizar

De los 57 flags disponibles en `posop_raw.parquet`, se aplican dos filtros:

**Filtro 1 — Exclusión por tipo de flag.** Se excluyen 20 flags que describen técnica, proceso o decisiones intraoperatorias, no complicaciones clínicas:

| Flag excluido | Razón de exclusión |
|---------------|-------------------|
| `flag_cancelacion` | Describe cancelación de procedimiento, no complicación clínica |
| `flag_complicacion` | Flag agregado — superset de otros flags |
| `flag_control_manual` | Decisión técnica de ventilación |
| `flag_desenlace` | Flag agregado de desenlace |
| `flag_estancia` | Superset de flags de estancia |
| `flag_fisiologicas` | Superset de flags fisiológicos |
| `flag_frecuencia_resp_anormal` | Parámetro ventilatorio intraop |
| `flag_hemoderivados` | Superset de reservas |
| `flag_hoja_laringoscopio_recta` | Decisión técnica del anestesiólogo |
| `flag_induccion` | Superset de flags de inducción |
| `flag_modos_avanzados` | Modo ventilatorio avanzado — decisión técnica |
| `flag_monitoreo_invasivo` | Decisión de monitoreo — no complicación |
| `flag_no_despierto` | Estado al salir de quirófano |
| `flag_parametros_ventilatorios` | Parámetros ventilatorios — técnica |
| `flag_reservas` | Superset de reservas de sangre |
| `flag_tecnica` | Flag técnico agregado |
| `flag_tecnica_combinada` | Técnica anestésica combinada — decisión |
| `flag_tiempos` | Duración quirúrgica — parcialmente controlable |
| `flag_ventilacion` | Superset de ventilación |
| `flag_ventilacion_asistida` | Modo ventilatorio — decisión técnica |

Quedan **37 flags candidatos** de carácter clínico o de resultado.

**Filtro 2 — Prevalencia mínima.** Se excluyen flags con prevalencia < 1% en la población de análisis (menos de ~234 casos), porque con tan pocos positivos la validación cruzada no es confiable estadísticamente. Los 11 flags excluidos por baja prevalencia son:

`flag_reserva_hemoderivados`, `flag_reserva_sangre`, `flag_elemento_via_aerea_complejo`, `flag_volumen_alto`, `flag_perdidas_altas`, `flag_infarto_miocardio`, `flag_acv_hemorragico`, `flag_tia`, `flag_aspiracion_pulmonar`, `flag_complicaciones_pulmonares`, `flag_complicaciones_medicas`.

Resultado final: **26 flags analizados**.

### 2.3 Modelo y validación

Para cada uno de los 26 flags, se entrena un **Random Forest Classifier** independiente:

```
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=1,
)
```

Se usa **validación cruzada estratificada de 5 folds** con `cross_val_score` y métrica `roc_auc`. La estratificación garantiza que cada fold tenga la misma proporción de positivos que el dataset completo, lo cual es importante dado que muchos flags tienen prevalencia baja.

Las **features de entrada** (variables preoperatorias) son todas las columnas de `merged.parquet` que no sean flags ni el target. Se aplica:
- Conversión a numérico (`pd.to_numeric`, errores → NaN)
- Eliminación de columnas con más del 60% de valores faltantes
- Imputación de faltantes con la mediana de cada columna

### 2.4 Clasificación de la señal

Cada flag se clasifica según su AUC promedio en CV:

| Categoría | AUC | Interpretación |
|-----------|-----|----------------|
| `buena_senal` | ≥ 0.75 | Las variables preop predicen bien este evento |
| `senal_moderada` | 0.65 – 0.74 | Hay señal, pero limitada |
| `senal_debil` | 0.55 – 0.64 | Señal marginal, casi aleatorio |
| `sin_senal` | < 0.55 | Las variables preop no predicen este evento |

---

## 3. Resultados completos

### 3.1 Resumen por categoría

| Categoría | Cantidad de flags |
|-----------|------------------|
| `buena_senal` | **7** |
| `senal_moderada` | **13** |
| `senal_debil` | **6** |
| `sin_senal` | **0** |

Todos los flags analizados tienen al menos señal débil (AUC > 0.55). Ninguno es completamente aleatorio desde las variables preoperatorias. Sin embargo, hay diferencias enormes entre el mejor (AUC 0.836) y el peor (AUC 0.552).

---

### 3.2 Flags con BUENA señal (AUC ≥ 0.75)

Estos flags son predecibles con fiabilidad desde los datos de la valoración preanestésica.

---

#### `flag_interconsultas` — AUC 0.836 ± 0.007

- **¿Qué es?** El paciente requirió interconsulta a otra especialidad médica durante o después de la cirugía.
- **Prevalencia:** 4.55% (1,065 pacientes de 23,387)
- **AUC:** 0.8359 ± 0.0072
- **Interpretación:** Es el flag más predecible de todos. Que un paciente vaya a necesitar interconsulta posoperatoria tiene sentido que sea predecible: las comorbilidades del paciente (enfermedades cardiovasculares, renales, hematológicas) que se registran en la valoración preanestésica son precisamente las que generan la necesidad de consultar a otra especialidad. Un paciente con insuficiencia renal crónica registrada en preop es muy probable que necesite interconsulta a nefrología si su función renal se ve afectada por la anestesia o el estrés quirúrgico.
- **Relevancia para el target:** Este flag tiene alta predictibilidad Y alto impacto clínico. Es un fuerte candidato para incluir en una definición de target más precisa.

---

#### `flag_presion_sistolica_anormal` — AUC 0.815 ± 0.009

- **¿Qué es?** La tensión arterial sistólica tuvo valores anormales durante el perioperatorio.
- **Prevalencia:** 4.67% (1,092 pacientes)
- **AUC:** 0.8149 ± 0.0093
- **Interpretación:** La tensión arterial sistólica preoperatoria es una de las variables más registradas en la valoración preanestésica. Tiene sentido que los pacientes con HTA mal controlada, o con valores basales alterados, sean los que muestren labilidad tensional durante la anestesia. La valoración preanestésica captura el estado cardiovascular basal del paciente, que es el principal determinante de la respuesta hemodinámica intraoperatoria.
- **Relevancia para el target:** Alta predictibilidad. Asociado directamente a riesgo cardiovascular perioperatorio. Clínicamente relevante.

---

#### `flag_hospitalizacion_no_anticipada` — AUC 0.814 ± 0.007

- **¿Qué es?** El paciente requirió hospitalización que no estaba planificada antes de la cirugía.
- **Prevalencia:** 19.39% (4,534 pacientes) — el flag más frecuente con buena señal.
- **AUC:** 0.8138 ± 0.0068
- **Interpretación:** Aproximadamente 1 de cada 5 pacientes tuvo una hospitalización no planificada. Este es probablemente el flag de mayor impacto clínico y administrativo: representa pacientes que llegaron con expectativa ambulatoria y terminaron hospitalizados, o cuya hospitalización se extendió más de lo previsto. Que sea predecible tiene sentido — la decisión de hospitalizar a un paciente inesperadamente depende en gran medida de sus comorbilidades, edad, estado general y complejidad del procedimiento, todo capturado en preop.
- **Relevancia para el target:** Muy alta prevalencia, muy alta predictibilidad, muy alto impacto. Este flag por sí solo podría ser el núcleo de un target redefinido.

---

#### `flag_uci_no_planeada` — AUC 0.776 ± 0.048

- **¿Qué es?** El paciente requirió ingreso a UCI que no estaba planificado.
- **Prevalencia:** 1.86% (434 pacientes)
- **AUC:** 0.7762 ± 0.0478
- **Nota:** La desviación estándar es relativamente alta (0.048) comparada con otros flags. Esto se debe a la baja prevalencia (434 casos en 5 folds = ~87 positivos por fold). La estimación es estadísticamente más ruidosa.
- **Interpretación:** El ingreso no planificado a UCI representa una de las complicaciones más graves. Los pacientes que la sufren tienen perfiles de alto riesgo que en principio son detectables preoperatoriamente: edad avanzada, múltiples comorbilidades, procedimientos de alta complejidad.
- **Relevancia para el target:** Aunque poco prevalente, es un evento de altísimo impacto clínico y predecible. Combinado con `flag_hospitalizacion_no_anticipada` representaría un target de "evento adverso mayor no anticipado".

---

#### `flag_estancia_uci` — AUC 0.776 ± 0.048

- **¿Qué es?** El paciente tuvo estancia en UCI (planeada o no).
- **Prevalencia:** 1.86% (434 pacientes)
- **AUC:** 0.7762 ± 0.0478
- **Nota:** AUC y prevalencia idénticos a `flag_uci_no_planeada`, lo que sugiere que ambos flags identifican casi exactamente los mismos pacientes. La distinción entre "UCI planeada" y "UCI no planeada" en los datos es más administrativa que clínica, pero ambas tienen la misma señal preoperatoria.
- **Interpretación:** Similar a `flag_uci_no_planeada`. Para fines del target, usar solo uno de los dos sería suficiente para evitar redundancia.

---

#### `flag_glucometria_anormal` — AUC 0.775 ± 0.020

- **¿Qué es?** La glucometría del paciente durante el perioperatorio presentó valores anormales (hipoglucemia o hiperglucemia significativa).
- **Prevalencia:** 4.03% (942 pacientes)
- **AUC:** 0.7754 ± 0.0197
- **Interpretación:** Este es un ejemplo paradigmático de predictibilidad esperada. Los pacientes diabéticos están registrados en la valoración preanestésica. La diabetes es el principal factor de riesgo para glucometría anormal perioperatoria. Las variables preop capturan exactamente el antecedente endocrinológico relevante. El modelo aprende que "diabetes registrada en preop → riesgo elevado de glucometría anormal".
- **Relevancia para el target:** Alta predictibilidad con mecanismo causal claro. Clínicamente relevante porque la glucometría anormal en perioperatorio se asocia a peores desenlaces postquirúrgicos.

---

#### `flag_estancia_prolongada` — AUC 0.762 ± 0.015

- **¿Qué es?** La estancia hospitalaria posoperatoria fue más prolongada de lo esperado para el procedimiento.
- **Prevalencia:** 3.65% (853 pacientes)
- **AUC:** 0.7622 ± 0.015
- **Interpretación:** La estancia prolongada está determinada por una combinación de factores: comorbilidades del paciente (que se capturan en preop), complejidad del procedimiento (también en preop) y complicaciones postquirúrgicas. El modelo puede predecir qué pacientes tienen mayor riesgo de estancia prolongada porque sus perfiles preoperatorios (edad avanzada, múltiples comorbilidades, procedimientos complejos) son los predictores naturales de esta variable.
- **Relevancia para el target:** Impacto alto desde perspectiva hospitalaria (días de cama, costos). Alta predictibilidad.

---

### 3.3 Flags con señal MODERADA (AUC 0.65 – 0.74)

Estos flags tienen señal real pero limitada. El modelo puede distinguir mejor que el azar entre quién va a presentarlos y quién no, pero con margen de error considerable.

---

#### `flag_presion_media_anormal` — AUC 0.740 ± 0.033

- **¿Qué es?** Presión arterial media anormal durante el perioperatorio.
- **Prevalencia:** 1.31% (306 pacientes)
- **Interpretación:** Similar a `flag_presion_sistolica_anormal` pero con menor prevalencia y señal algo más débil. La presión media integra sistólica y diastólica, siendo una medida más global del estado hemodinámico. La mayor desviación estándar (0.033) refleja la baja prevalencia.

---

#### `flag_duracion_cirujano_larga` — AUC 0.728 ± 0.017

- **¿Qué es?** La duración quirúrgica desde la perspectiva del cirujano fue excepcionalmente larga.
- **Prevalencia:** 8.40% (1,965 pacientes)
- **Interpretación:** La duración quirúrgica larga depende de la complejidad del procedimiento y del estado del paciente. Los procedimientos de alta complejidad (registrados en preop como scores de severidad del procedimiento) y los pacientes con comorbilidades que dificultan la cirugía son los que terminan con tiempos quirúrgicos extendidos. El modelo capta esto con señal moderada.

---

#### `flag_seguimiento` — AUC 0.720 ± 0.014

- **¿Qué es?** El paciente requirió seguimiento posoperatorio adicional o especial.
- **Prevalencia:** 8.60% (2,011 pacientes)
- **Interpretación:** La necesidad de seguimiento adicional refleja la complejidad del caso. Es esperable que pacientes de alto riesgo preoperatorio requieran más seguimiento posoperatorio.

---

#### `flag_laringoscopia_alta` — AUC 0.719 ± 0.008

- **¿Qué es?** La laringoscopia presentó clasificación alta (Cormack-Lehane III o IV), indicando vía aérea difícil.
- **Prevalencia:** 12.40% (2,899 pacientes)
- **Interpretación:** El Puntaje Mallampati es una variable preoperatoria directamente relacionada con la predicción de vía aérea difícil. Que este flag tenga señal moderada (no buena) sugiere que el Mallampati predice parcialmente la dificultad laringoscópica, pero hay factores intraoperatorios (posición del paciente, equipo disponible, experiencia del anestesiólogo) que el modelo no puede anticipar.

---

#### `flag_duracion_anestesia_larga` — AUC 0.718 ± 0.009

- **¿Qué es?** La duración total de la anestesia fue excepcionalmente larga.
- **Prevalencia:** 5.76% (1,347 pacientes)
- **Interpretación:** Similar a `flag_duracion_cirujano_larga`. La señal moderada (vs. buena) confirma que la duración anestésica tiene un componente predecible (complejidad del procedimiento, comorbilidades) pero también un componente impredecible (complicaciones intraoperatorias que alargan el procedimiento).

---

#### `flag_destino_uci` — AUC 0.712 ± 0.029

- **¿Qué es?** El destino posquirúrgico inmediato del paciente fue la UCI (independientemente de si estaba planeado).
- **Prevalencia:** 1.07% (250 pacientes)
- **Interpretación:** El destino UCI está determinado por la complejidad del procedimiento y el estado del paciente. Los pacientes de muy alto riesgo que llegan con destino UCI planeado hacen que este flag sea parcialmente predecible desde preop.

---

#### `flag_intubacion_dificil` — AUC 0.704 ± 0.020

- **¿Qué es?** La intubación orotraqueal fue difícil (más de 2 intentos, dispositivos alternativos, etc.).
- **Prevalencia:** 2.24% (525 pacientes)
- **Interpretación:** La intubación difícil es una de las principales preocupaciones de la valoración preanestésica — de hecho, el Puntaje Mallampati existe precisamente para predecirla. Que el AUC sea 0.70 (no más alto) tiene dos explicaciones: (1) el Mallampati es un predictor imperfecto con falsos negativos frecuentes, y (2) factores intraoperatorios como edema de vías aéreas, posición, o espasmo laríngeo son impredecibles.

---

#### `flag_saturacion_oxigeno_anormal` — AUC 0.695 ± 0.018

- **¿Qué es?** La saturación de oxígeno presentó valores anormales durante el perioperatorio.
- **Prevalencia:** 3.22% (752 pacientes)
- **Interpretación:** Los pacientes con antecedentes respiratorios (EPOC, asma) y obesidad tienen mayor riesgo de desaturación. Estas comorbilidades se capturan en preop. La señal moderada refleja que también hay causas intraoperatorias impredecibles (broncoespasmo agudo, obstrucción de vía aérea).

---

#### `flag_balance_extremo` — AUC 0.689 ± 0.030

- **¿Qué es?** El balance hídrico durante la cirugía fue extremo (ya sea balance positivo o negativo muy marcado).
- **Prevalencia:** 1.07% (251 pacientes)
- **Interpretación:** El balance hídrico extremo puede deberse a pérdidas quirúrgicas masivas (sangrado abundante) o a resucitación agresiva. Es parcialmente predecible por el tipo de procedimiento y el estado cardiovascular basal del paciente, pero también depende de eventos intraoperatorios.

---

#### `flag_via_aerea` — AUC 0.686 ± 0.003

- **¿Qué es?** Flag agregado que indica algún tipo de complicación o manejo complejo de vía aérea.
- **Prevalencia:** 15.38% (3,598 pacientes) — uno de los más frecuentes.
- **Nota:** La desviación estándar muy baja (0.003) indica estimación muy estable a pesar de la señal moderada. Con ~3,598 positivos, los 5 folds tienen suficiente representación.
- **Interpretación:** Al ser un flag agregado, mezcla la señal de múltiples eventos de vía aérea. La señal moderada refleja esta heterogeneidad.

---

#### `flag_temperatura_anormal` — AUC 0.670 ± 0.013

- **¿Qué es?** La temperatura del paciente presentó valores anormales durante el perioperatorio (hipotermia o fiebre).
- **Prevalencia:** 2.81% (658 pacientes)
- **Interpretación:** La hipotermia perioperatoria es más frecuente en pacientes con baja reserva metabólica (ancianos, desnutridos, bajo peso). La fiebre puede asociarse a infecciones previas o reacciones inflamatorias. Ambas tienen componentes predecibles desde el estado preoperatorio del paciente.

---

#### `flag_tipo_intubacion_complejo` — AUC 0.669 ± 0.014

- **¿Qué es?** Se utilizó un tipo de intubación no estándar o complejo (fibroscopia, intubación nasal, videolaringoscopia, etc.).
- **Prevalencia:** 2.22% (519 pacientes)
- **Interpretación:** Similar a `flag_intubacion_dificil`. El tipo de intubación complejo está parcialmente determinado por la anatomía del paciente (Mallampati, apertura bucal, extensión cervical) registrada en preop.

---

#### `flag_induccion_compleja` — AUC 0.663 ± 0.021

- **¿Qué es?** La inducción anestésica presentó complicaciones o requirió técnicas especiales.
- **Prevalencia:** 3.51% (821 pacientes)
- **Interpretación:** La inducción compleja puede deberse a vía aérea difícil anticipada, inestabilidad hemodinámica durante la inducción, o reacciones medicamentosas. La señal moderada-baja refleja que parte de esto es impredecible.

---

### 3.4 Flags con señal DÉBIL (AUC 0.55 – 0.64)

Estos flags son parcialmente impredecibles desde variables preoperatorias. El modelo distingue algo mejor que el azar, pero con poca fiabilidad clínica.

---

#### `flag_presion_diastolica_anormal` — AUC 0.619 ± 0.006

- **¿Qué es?** La tensión arterial diastólica presentó valores anormales.
- **Prevalencia:** 1.88% (440 pacientes)
- **Interpretación:** Curiosamente, la presión diastólica tiene señal más débil que la sistólica (0.619 vs. 0.815). Esto puede explicarse porque la variabilidad diastólica intraoperatoria depende en mayor medida de la vasodilatación farmacológica y el tono vascular, que son respuestas más variables e impredecibles que la presión sistólica.

---

#### `flag_liquidos` — AUC 0.617 ± 0.020

- **¿Qué es?** El volumen de líquidos administrado durante la cirugía fue elevado.
- **Prevalencia:** 3.74% (875 pacientes)
- **Interpretación:** La administración de líquidos es una decisión intraoperatoria del anestesiólogo en respuesta a los hallazgos en tiempo real. Aunque el tipo de procedimiento (predecible) influye, la cantidad exacta depende de factores intraoperatorios como el sangrado, la respuesta cardiovascular y el protocolo del anestesiólogo. Por eso la señal es débil.

---

#### `flag_aferesis` — AUC 0.595 ± 0.028

- **¿Qué es?** El paciente requirió aféresis (procedimiento de separación y remoción de componentes sanguíneos).
- **Prevalencia:** 2.63% (614 pacientes)
- **Interpretación:** La aféresis es un procedimiento altamente especializado, típicamente para pacientes hematológicos o con trastornos específicos. Su baja señal (0.595) sugiere que los pacientes que la requieren no siempre tienen un perfil preoperatorio claramente diferenciado del resto, o que la indicación emerge intraoperatoriamente de manera no anticipada.

---

#### `flag_urgencias_30_dias` — AUC 0.592 ± 0.025

- **¿Qué es?** El paciente acudió a urgencias en los 30 días posteriores a la cirugía.
- **Prevalencia:** 4.37% (1,022 pacientes)
- **Interpretación:** Las visitas a urgencias posoperatorias son el evento más difícil de predecir. Dependen de complicaciones tardías, infecciones, dolor mal controlado, o re-sangrado — eventos que ocurren semanas después de la cirugía y cuya causa puede ser tan impredecible como una infección del sitio quirúrgico. Que tenga señal débil (no nula) sugiere que los pacientes más vulnerables (multimorbilidad, bajo nivel socioeconómico) tienen mayor riesgo, pero la variabilidad es enorme.

---

#### `flag_frecuencia_cardiaca_anormal` — AUC 0.584 ± 0.018

- **¿Qué es?** La frecuencia cardíaca presentó valores anormales (taquicardia o bradicardia significativa) durante el perioperatorio.
- **Prevalencia:** 1.44% (337 pacientes)
- **Interpretación:** Las arritmias perioperatorias dependen de múltiples factores intraoperatorios: anestésicos (los agentes halogenados son arritmogénicos), estimulación quirúrgica, dolor, hipoxia. Aunque los pacientes con cardiopatías basales tienen mayor riesgo, el desencadenante específico es intraoperatorio e impredecible desde preop.

---

#### `flag_intubado_salida` — AUC 0.552 ± 0.011

- **¿Qué es?** El paciente salió de quirófano intubado (con tubo endotraqueal, sin extubación al final del procedimiento).
- **Prevalencia:** 11.53% (2,696 pacientes) — uno de los más frecuentes.
- **AUC:** 0.5520 — el más bajo de todos los analizados.
- **Interpretación:** Este es el caso más claro de evento impredecible desde preop. La decisión de extubar o no al paciente al final de la cirugía es una decisión médica en tiempo real que depende de: el curso intraoperatorio, el tipo de anestesia usada, la estabilidad hemodinámica, la capacidad ventilatoria al momento de la reversión, y el criterio del anestesiólogo. Un AUC de 0.55 significa que el modelo no puede distinguir consistentemente qué pacientes van a salir intubados desde los datos preoperatorios — lo cual es esperable porque esa información simplemente no está disponible antes de la cirugía.
- **Implicación:** Si este flag está incluido en el target compuesto, está añadiendo ruido puro al modelo. Cada paciente que salió intubado "por razones intraoperatorias normales" que el modelo clasifica como falso negativo está degradando las métricas sin que haya manera de aprenderlo.

---

## 4. Análisis comparativo e implicaciones

### 4.1 ¿Qué determina si un flag es predecible?

El patrón es claro al observar los resultados:

**Alta predictibilidad** (AUC ≥ 0.75):
- El evento tiene un mecanismo causal directo con variables registradas en preop.
- Ejemplo: diabetes preop → glucometría anormal (causalidad directa).
- Ejemplo: comorbilidades múltiples → hospitalización no anticipada (causalidad acumulada).

**Señal moderada** (AUC 0.65–0.74):
- El evento depende parcialmente de variables preop y parcialmente de factores intraoperatorios.
- Ejemplo: Mallampati → intubación difícil (predictor imperfecto, hay factores intraop).

**Señal débil** (AUC 0.55–0.64):
- El evento está dominado por factores intraoperatorios o posoperatorios tardíos.
- Ejemplo: salida intubado (decisión intraoperatoria pura).

### 4.2 El problema del target compuesto actual

El target `target_d_v2_hosp` es un OR lógico de múltiples flags. Si incluye tanto `flag_hospitalizacion_no_anticipada` (AUC individual 0.81) como `flag_intubado_salida` (AUC individual 0.55), el modelo debe aprender simultáneamente dos señales incompatibles:

- Para `hospitalizacion_no_anticipada`: "pacientes con comorbilidades tienen mayor riesgo → positivo"
- Para `intubado_salida`: "no hay patrón claro desde preop → prácticamente aleatorio"

El resultado es que la señal de los eventos predecibles queda diluida por el ruido de los impredecibles.

### 4.3 Propuesta de target redefinido

Basado en este análisis, un target más predecible podría construirse solo con los flags de buena señal que representan eventos de alto impacto clínico:

```
target_redefinido = (
    flag_hospitalizacion_no_anticipada OR
    flag_uci_no_planeada OR
    flag_estancia_uci OR
    flag_interconsultas OR
    flag_estancia_prolongada OR
    flag_glucometria_anormal
)
```

Este target representaría "eventos adversos posoperatorios mayores, de alto impacto clínico, predecibles desde la valoración preanestésica". Su prevalencia estimada sería mayor que la de cualquier flag individual pero menor que el target actual (que suma todo), y su AUC en un nuevo modelo podría superar 0.80.

---

## 5. Resumen de resultados en tabla completa

| Flag | Prevalencia | N positivos | AUC medio | AUC std | Categoría |
|------|-------------|-------------|-----------|---------|-----------|
| `flag_interconsultas` | 4.55% | 1,065 | **0.8359** | 0.0072 | buena_senal |
| `flag_presion_sistolica_anormal` | 4.67% | 1,092 | **0.8149** | 0.0093 | buena_senal |
| `flag_hospitalizacion_no_anticipada` | 19.39% | 4,534 | **0.8138** | 0.0068 | buena_senal |
| `flag_uci_no_planeada` | 1.86% | 434 | **0.7762** | 0.0478 | buena_senal |
| `flag_estancia_uci` | 1.86% | 434 | **0.7762** | 0.0478 | buena_senal |
| `flag_glucometria_anormal` | 4.03% | 942 | **0.7754** | 0.0197 | buena_senal |
| `flag_estancia_prolongada` | 3.65% | 853 | **0.7622** | 0.0150 | buena_senal |
| `flag_presion_media_anormal` | 1.31% | 306 | 0.7404 | 0.0329 | senal_moderada |
| `flag_duracion_cirujano_larga` | 8.40% | 1,965 | 0.7277 | 0.0174 | senal_moderada |
| `flag_seguimiento` | 8.60% | 2,011 | 0.7203 | 0.0143 | senal_moderada |
| `flag_laringoscopia_alta` | 12.40% | 2,899 | 0.7190 | 0.0081 | senal_moderada |
| `flag_duracion_anestesia_larga` | 5.76% | 1,347 | 0.7183 | 0.0090 | senal_moderada |
| `flag_destino_uci` | 1.07% | 250 | 0.7122 | 0.0289 | senal_moderada |
| `flag_intubacion_dificil` | 2.24% | 525 | 0.7044 | 0.0195 | senal_moderada |
| `flag_saturacion_oxigeno_anormal` | 3.22% | 752 | 0.6947 | 0.0178 | senal_moderada |
| `flag_balance_extremo` | 1.07% | 251 | 0.6892 | 0.0302 | senal_moderada |
| `flag_via_aerea` | 15.38% | 3,598 | 0.6862 | 0.0034 | senal_moderada |
| `flag_temperatura_anormal` | 2.81% | 658 | 0.6702 | 0.0127 | senal_moderada |
| `flag_tipo_intubacion_complejo` | 2.22% | 519 | 0.6692 | 0.0141 | senal_moderada |
| `flag_induccion_compleja` | 3.51% | 821 | 0.6625 | 0.0207 | senal_moderada |
| `flag_presion_diastolica_anormal` | 1.88% | 440 | 0.6188 | 0.0060 | senal_debil |
| `flag_liquidos` | 3.74% | 875 | 0.6174 | 0.0196 | senal_debil |
| `flag_aferesis` | 2.63% | 614 | 0.5950 | 0.0276 | senal_debil |
| `flag_urgencias_30_dias` | 4.37% | 1,022 | 0.5924 | 0.0251 | senal_debil |
| `flag_frecuencia_cardiaca_anormal` | 1.44% | 337 | 0.5835 | 0.0175 | senal_debil |
| `flag_intubado_salida` | 11.53% | 2,696 | 0.5520 | 0.0110 | senal_debil |
