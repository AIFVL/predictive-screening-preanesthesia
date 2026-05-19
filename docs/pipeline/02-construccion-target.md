# Etapa 2 — Construcción de la Variable Objetivo (Target)

**Código fuente:**
- [`src/target/builder.py`](../../src/target/builder.py) — Constructor genérico del target por threshold de flags
- [`src/target/pipeline.py`](../../src/target/pipeline.py) — Pipeline completo de validación y extracción
- [`src/target/constants.py`](../../src/target/constants.py) — Definición de flags relevantes y excluidos
- [`src/target/specific/`](../../src/target/specific/) — Módulos de cálculo de cada flag individual
- [`config/target_config.yaml`](../../config/target_config.yaml) — Configuración de todas las versiones del target

**Outputs (pipeline v2):**
- `output/v2/data_processed/{target}/target_extracted.parquet` — Dataset posoperatorio con la columna `target` añadida
- `output/v2/data_processed/{target}/merged.parquet` — Tras el join con preop, dataset listo para selección de features

---

## 1. El problema de definir el target

La pregunta central del proyecto es: *"¿Este paciente necesita valoración preanestésica formal?"*. La respuesta esperada sería afirmativa para aquellos pacientes que presentaron complicaciones o eventos adversos durante o después de la cirugía. Sin embargo, el dataset posoperatorio registra decenas de eventos de naturaleza y severidad muy distintas, lo que convierte la definición del target en una decisión clínica y estadística de primer orden. Un target mal definido obliga al modelo a aprender patrones imposibles o ruidosos, comprometiendo directamente su capacidad predictiva.

### El problema del target compuesto

Un target construido como la unión lógica (OR) de múltiples flags tiene dos efectos contrapuestos. Por un lado, incrementa la prevalencia al incorporar más casos positivos, lo que favorece el aprendizaje. Por otro lado, si incluye flags con escasa predictibilidad desde variables preoperatorias, introduce ruido que deteriora la señal del target. El análisis del Enfoque C ([ver](../analisis-posoperatorio/enfoque-C-flag-predictability.md)) demostró empíricamente que algunos flags son altamente predecibles desde preop (AUC individual > 0.80) mientras que otros son casi informativamente nulos (AUC ~ 0.55). El análisis del Enfoque A ([ver](../analisis-posoperatorio/enfoque-A-posop-clustering.md)) reveló además la existencia de fenotipos de complicación clínicamente distintos que el modelo actual trata de forma indiferenciada.

---

## 2. Versiones del target evaluadas

A lo largo del proyecto se han evaluado **9 versiones** distintas del target (a, b, c, d, d_v2, d_v2_hosp, d_v5, e, f). Las **dos versiones activas** en el pipeline v2 (campo `active_targets` en [`config/target_config.yaml`](../../config/target_config.yaml)) son:

- `target_d_v2_hosp` — versión histórica, conservada como referencia comparativa.
- `target_f_predictibilidad_maxima` — **versión recomendada** y servida por defecto en la API.

Las versiones c, e y antiguas iteraciones de d siguen documentadas en `target_config.yaml` pero no se entrenan en cada corrida del pipeline.

### `target_d_v2` — Target refinado sin hospitalización

**Prevalencia:** 16.93% (3,961 positivos de 23,387)

**Lógica de construcción (subflag_logic):**
```
flag_via_aerea_r  = flag_intubacion_dificil OR flag_tipo_intubacion_complejo OR flag_elemento_via_aerea_complejo
flag_estancia_r   = flag_uci_no_planeada OR flag_estancia_uci
flag_desenlace_r  = flag_destino_uci OR (flag_intubado_salida AND flag_uci_no_planeada)

target = flag_via_aerea_r OR flag_estancia_r OR flag_desenlace_r
       OR flag_complicaciones_medicas OR flag_liquidos OR flag_seguimiento
```

**Componentes y su prevalencia en posop_raw (29,865 registros):**

| Componente del target | Descripción | N | Prevalencia |
|----------------------|-------------|---|-------------|
| `flag_intubacion_dificil` | Intubación difícil | 544 | 1.8% |
| `flag_tipo_intubacion_complejo` | Tipo de intubación complejo | 817 | 2.7% |
| `flag_elemento_via_aerea_complejo` | Dispositivo de vía aérea complejo | 76 | 0.3% |
| `flag_uci_no_planeada` | UCI no planificada | 670 | 2.2% |
| `flag_estancia_uci` | Estancia en UCI | 670 | 2.2% |
| `flag_destino_uci` | Destino: UCI | 379 | 1.3% |
| `flag_intubado_salida AND flag_uci_no_planeada` | Sale intubado + UCI | ~200 | ~0.7% |
| `flag_complicaciones_medicas` | Complicaciones médicas | 198 | 0.7% |
| `flag_liquidos` | Líquidos elevados | 986 | 3.3% |
| `flag_seguimiento` | Requirió seguimiento | 2,339 | 7.8% |

Este target fue diseñado para incluir eventos clínicamente relevantes y excluir aquellos que representan decisiones intraoperatorias sin consecuencia directa para el paciente. La lógica de `flag_desenlace_r` es especialmente cuidadosa: salir intubado solo se computa como evento si se combina con UCI no planificada, evitando así contabilizar las intubaciones propias de procedimientos de alta complejidad previamente planificados.

Su principal limitación es la prevalencia de 16.93%, que genera un dataset muy desbalanceado. Además, los modelos entrenados sobre este target alcanzan AUC ~0.64, notablemente inferior al de otras versiones, lo que sugiere que la señal disponible en esos flags es insuficiente para el aprendizaje.

---

### `target_d_v2_hosp` — Target refinado + hospitalización no anticipada

**Prevalencia:** 27.69% (6,475 positivos de 23,387)

**Lógica de construcción:**
```
flag_via_aerea_r  = flag_intubacion_dificil OR flag_tipo_intubacion_complejo OR flag_elemento_via_aerea_complejo
flag_estancia_r   = flag_uci_no_planeada OR flag_estancia_uci
flag_desenlace_r  = flag_destino_uci OR (flag_intubado_salida AND flag_uci_no_planeada)

target = flag_via_aerea_r OR flag_estancia_r OR flag_desenlace_r
       OR flag_complicaciones_medicas OR flag_liquidos OR flag_seguimiento
       OR flag_hospitalizacion_no_anticipada   ← nuevo componente
```

**El componente añadido — `flag_hospitalizacion_no_anticipada`:**
- **N positivos:** 5,273 de 29,865 (17.7% del posop total)
- Es el flag individual más prevalente y también el **segundo más predecible** desde preop (AUC 0.814)
- Representa un evento de alto impacto clínico y administrativo: pacientes que llegarían ambulatorios pero terminan hospitalizados

La adición de este flag responde a dos motivaciones concretas. Primero, eleva la prevalencia de 16.93% a 27.69%, lo que le entrega al modelo más señal positiva para aprender. Segundo, mejora el AUC de ~0.64 a ~0.75, dado que la hospitalización no anticipada es el flag individual con mayor predictibilidad desde variables preoperatorias. Esta versión fue relevante en el proceso de selección por su mejor balance entre prevalencia manejable, interpretabilidad clínica y rendimiento del modelo respecto a iteraciones anteriores. Los resultados del Enfoque C respaldan incluir `flag_hospitalizacion_no_anticipada` como componente central de cualquier redefinición del target.

---

### `target_f_predictibilidad_maxima` — Target de máxima predictibilidad *(versión recomendada en v2)*

**Prevalencia:** 19.43% (~4,544 positivos de 23,387)

**Lógica de construcción:**
```
target = flag_interconsultas
       OR flag_hospitalizacion_no_anticipada
       OR flag_uci_no_planeada
       OR flag_estancia_uci
       OR flag_estancia_prolongada
```

**Motivación:** Después de `target_d_v2_hosp` se invirtió el enfoque de diseño. En lugar de seleccionar primero los eventos clínicamente relevantes y aceptar el rendimiento resultante, este target se construye seleccionando los flags con mayor predictibilidad desde variables preoperatorias — mayor AUC individual y mayor información mutua con las features preop — bajo la premisa de que esos son, por construcción, los eventos que un modelo de screening puede detectar de forma confiable.

Los cinco flags incluidos son los que el [análisis del Enfoque C](../analisis-posoperatorio/enfoque-C-flag-predictability.md) identificó como los más predecibles individualmente. Todos son eventos clínicamente relevantes:

| Flag | Naturaleza clínica |
|------|---------------------|
| `flag_interconsultas` | Necesidad de interconsulta especializada — indica complejidad imprevista |
| `flag_hospitalizacion_no_anticipada` | Paciente ambulatorio que termina hospitalizado |
| `flag_uci_no_planeada` | Ingreso a UCI no planificado |
| `flag_estancia_uci` | Estancia en UCI (planeada o no) |
| `flag_estancia_prolongada` | Estancia hospitalaria > 3 días sobre lo esperado |

**Rendimiento:** AUC en test ~0.86 con XGBoost, vs. ~0.76 para `target_d_v2_hosp`. Es la versión con mayor señal disponible en el pipeline.

**Análisis comparativo de señal preop→target** (de [`output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv`](../../output/v2/reports/pre_post_signal/pre_post_linkage_summary.csv)):

| Versión target | Prevalencia | Max MI | Max Pearson | N features informativas |
|---|---|---|---|---|
| `target_d_v2_hosp` | 27.69% | 0.100 | 0.232 | 16 |
| `target_f_predictibilidad_maxima` | 19.43% | **0.130** | **0.303** | 16 |

A pesar de la menor prevalencia (19.43% vs. 27.69%), la señal disponible es genuinamente superior: todos los modelos incrementan su AUC de ~0.76 a ~0.86 al cambiar al target F. Por esta razón es el target servido por defecto en la API, donde el `TargetAlias` con `slug="hospitalization_risk"` en [`api/core/config.py`](../../api/core/config.py) mapea a `target_f_predictibilidad_maxima` con `recommended=True`.

---

### `target_d_v5` — Target de alta severidad *(legacy, solo en v1)*

**Prevalencia (v1):** 25.63% (5,997 positivos de 23,387)

**Lógica de construcción:**
```
target = flag_destino_uci OR flag_uci_no_planeada OR flag_complicaciones_medicas
       OR flag_intubacion_dificil OR flag_aferesis OR flag_perdidas_altas
       OR flag_urgencias_30_dias OR flag_hospitalizacion_no_anticipada
```

A diferencia de `target_d_v2_hosp`, esta versión priorizaba eventos de mayor severidad: incluye `flag_urgencias_30_dias` y `flag_aferesis`, mientras excluye `flag_seguimiento`, `flag_liquidos` y `flag_tipo_intubacion_complejo` por considerarlos de severidad moderada, y omite `flag_estancia_uci` por ser redundante con `flag_uci_no_planeada`. Su AUC (~0.76) es similar al de `target_d_v2_hosp`, por lo que fue desactivado en v2 al quedar superado por `target_f_predictibilidad_maxima` (~0.86) con una composición más simple.

---

## 3. Versiones históricas (archivadas)

### `target_a_sensible` — Target amplio

```
target = OR de todos los flags (excepto excluidos)
```
Incluía todos los flags: cancelaciones, reservas, fisiológicos, tiempos, inducción, vía aérea, ventilación, técnica, líquidos, desenlace, complicaciones médicas, estancia, seguimiento. Resultaba en prevalencias >50%, haciendo el problema trivial y con mala señal real.

### `target_b_clinicamente_relevante`

```
target = flag_cancelacion OR flag_reservas OR flag_via_aerea OR flag_induccion
       OR flag_tiempos OR flag_tecnica OR flag_complicaciones_medicas
       OR flag_estancia OR flag_seguimiento
```
Excluía ventilación y fisiológicos. Prevalencia ~35%, mejor que A pero con mucho ruido de flags de técnica.

### `target_c_alta_severidad`

Mismo conjunto que B pero con threshold ≥ 2 flags (el paciente debe tener al menos 2 flags para ser positivo). Intentaba reducir falsos positivos, pero reducía también la prevalencia a niveles muy bajos.

### `target_d_eventos_adversos`

Primera iteración de D, sin la subflag_logic refinada. Precedente directo de `target_d_v2`.

### `target_e_alta_senal`

Intento de definir el target solo con los flags con MI > 0.01 desde preop (`flag_estancia`, `flag_tecnica`, `flag_tiempos`, `flag_fisiologicas`). Sirvió de antecesor metodológico de `target_f_predictibilidad_maxima`, pero con peor selección de flags.

---

## 4. Regla de exclusión de cancelaciones no-médicas

Una regla especial se aplica uniformemente en todas las versiones del target:

```python
if apply_cancel_non_medico_rule:
    mask = (df["canceladas"] == 1) & (df["canceladas_por_medico"] == 0)
    df.loc[mask, "target"] = 0
```

Un procedimiento cancelado por razones no-médicas — el paciente no se presentó, cancelación administrativa, indisponibilidad de sala — no constituye evidencia de que el paciente requiera valoración preanestésica, puesto que la cancelación no se origina en un hallazgo clínico. Mantener estos casos como positivos llevaría al modelo a aprender patrones de cancelación administrativa en lugar de riesgo perioperatorio real. Solo las cancelaciones motivadas por un hallazgo clínico que contraindica el procedimiento son relevantes para el target.

---

## 5. Distribución del target en las versiones activas

### `target_d_v2_hosp`
```
Total pacientes en merged: 23,387
├── Negativos (target=0): 16,912 (72.3%)
└── Positivos (target=1):  6,475 (27.7%)

División train/test (80/20 estratificado, random_state=42):
├── Train: 18,709 | 5,180 positivos (27.7%)
└── Test:   4,678 | 1,295 positivos (27.7%)
```

### `target_f_predictibilidad_maxima` *(recomendado)*
```
Total pacientes en merged: 23,387
├── Negativos (target=0): ~18,843 (80.57%)
└── Positivos (target=1):  ~4,544 (19.43%)

División train/test (80/20 estratificado, random_state=42):
├── Train: 18,709 registros (mismo split que d_v2_hosp)
└── Test:   4,678 registros
```

La estratificación garantiza que el desbalance de clases sea idéntico en los conjuntos de entrenamiento y prueba, evitando que una partición aleatoria desfavorable sesgue la evaluación. Ambos targets se entrenan sobre el mismo split de pacientes; la única diferencia entre ellos es el contenido de la columna `target`.

---

## 6. ¿Qué queda fuera del target y por qué?

Los siguientes flags fueron **excluidos** de todas las versiones del target en uso:

| Flag excluido | Razón |
|---------------|-------|
| `flag_fisiologicas` | Superset de flags fisiológicos — redundante con los específicos |
| `flag_ventilacion` | Decisiones de ventilación son técnicas, no complicaciones clínicas |
| `flag_tecnica` | Técnica anestésica combinada/monitoreo invasivo: decisión clínica planeada |
| `flag_tiempos` | Duración quirúrgica larga puede ser cirugía compleja prevista, no complicación |
| `flag_induccion` | Inducción compleja: elección técnica del anestesiólogo |
| `flag_reservas` | Reservar sangre es precaución, no complicación — la transfusión real está en `flag_liquidos` |
| `flag_cancelacion` | Solo incluida en versiones antiguas; en versiones actuales se usa para la regla de exclusión |

Adicionalmente, `flag_intubado_salida` está en el target v2 solo cuando se combina con `flag_uci_no_planeada`, porque salir intubado solo (sin UCI) es frecuentemente una decisión planificada, no una complicación.

---

## 7. Implicaciones del análisis posoperatorio sobre el target

Los tres enfoques del [análisis posoperatorio](../analisis-posoperatorio/README.md) aportan evidencia convergente sobre cómo mejorar la definición del target. El Enfoque C identifica `flag_hospitalizacion_no_anticipada` (AUC 0.814), `flag_interconsultas` (0.836) y `flag_glucometria_anormal` (0.775) como los flags más predecibles individualmente desde variables preoperatorias; el target actual incorpora el primero pero no los otros dos. El Enfoque A, basado en clustering, muestra que los pacientes del Cluster C2 — definidos por `flag_intubado_salida` — tienen perfiles preoperatorios casi indistinguibles de los negativos, confirmando que este flag no debe incluirse en el target salvo cuando coincide con UCI no planificada.

La convergencia de estos análisis se materializó en `target_f_predictibilidad_maxima`:
   ```
   target_f = flag_interconsultas
            OR flag_hospitalizacion_no_anticipada
            OR flag_uci_no_planeada
            OR flag_estancia_uci
            OR flag_estancia_prolongada
   ```
Esta composición recoge cinco de los seis flags propuestos por el Enfoque C con mayor AUC individual, descartando `flag_glucometria_anormal` por dudas sobre la calidad de su registro. El resultado es un AUC ~0.86 en test, frente al ~0.76 de `target_d_v2_hosp`.
