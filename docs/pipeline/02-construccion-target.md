# Etapa 2 — Construcción de la Variable Objetivo (Target)

**Código fuente:**
- [`src/target/builder.py`](../../src/target/builder.py) — Constructor genérico del target por threshold de flags
- [`src/target/pipeline.py`](../../src/target/pipeline.py) — Pipeline completo de validación y extracción
- [`src/target/constants.py`](../../src/target/constants.py) — Definición de flags relevantes y excluidos
- [`src/target/specific/`](../../src/target/specific/) — Módulos de cálculo de cada flag individual
- [`config/target_config.yaml`](../../config/target_config.yaml) — Configuración de todas las versiones del target

**Outputs:**
- `output/v1/data_processed/{target}/merged.parquet` — Dataset con columna `target` añadida

---

## 1. El problema de definir el target

La pregunta del proyecto es: *"¿Este paciente necesita valoración preanestésica formal?"*

La respuesta debería ser: "sí, si presentó complicaciones o eventos adversos en la cirugía". Pero ¿qué cuenta como complicación? El dataset posoperatorio registra docenas de eventos distintos, con naturaleza y severidad muy diferentes. Definir el target implica tomar decisiones clínicas y estadísticas sobre cuáles eventos son relevantes.

Esta es la decisión más crítica del proyecto, porque afecta directamente qué señal tiene el modelo para aprender. Un target mal definido hace que el modelo intente aprender patrones imposibles o ruidosos.

### El problema del target compuesto

Un target que combina muchos flags distintos con OR lógico tiene dos efectos opuestos:
1. **Aumenta la prevalencia** (más positivos), lo que ayuda al aprendizaje.
2. **Reduce la señal** si incluye flags que no son predecibles desde variables preoperatorias — añade ruido al target.

El análisis del Enfoque C ([ver](../analisis-posoperatorio/enfoque-C-flag-predictability.md)) demostró empíricamente que algunos flags son muy predecibles desde preop (AUC individual > 0.80) mientras que otros son casi aleatorios (AUC ~ 0.55). El análisis del Enfoque A ([ver](../analisis-posoperatorio/enfoque-A-posop-clustering.md)) mostró que hay fenotipos de complicación clínicamente distintos que el modelo actual trata como uno solo.

---

## 2. Versiones del target evaluadas

Se evaluaron múltiples definiciones del target. Las tres versiones activas en el pipeline son:

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

**¿Por qué este target?** Se diseñó para incluir eventos clínicamente relevantes pero excluir los que son solo decisiones intraoperatorias sin consecuencia clínica para el paciente. La lógica de `flag_desenlace_r` es especialmente cuidadosa: salir intubado solo cuenta si además hubo UCI no planificada — evita contar las intubaciones planificadas de procedimientos de alta complejidad.

**Problema:** La prevalencia del 16.93% resulta en un dataset muy desbalanceado. Además, los modelos logran AUC ~0.64, notablemente inferior a otras versiones. Esto sugiere que la señal contenida en esos flags es insuficiente para el aprendizaje.

---

### `target_d_v2_hosp` — Target refinado + hospitalización no anticipada *(versión seleccionada)*

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

**¿Por qué se añadió?** Dos razones principales:
1. **Mejora de prevalencia:** 16.93% → 27.69%. Con más positivos, el modelo tiene más señal para aprender.
2. **Mejora de AUC:** Los modelos mejoran de ~0.64 a ~0.75. La hospitalización no anticipada es el flag con mayor señal individual desde preop, y añadirla al target mejora la señal total.

**¿Por qué es la versión seleccionada?** Mejor balance entre prevalencia manejable, interpretabilidad clínica y rendimiento del modelo. Los resultados del Enfoque C apoyan incluir `flag_hospitalizacion_no_anticipada` como componente central de cualquier target redefinido.

---

### `target_d_v5` — Target de alta severidad

**Prevalencia:** 25.63% (5,997 positivos de 23,387)

**Lógica de construcción:**
```
target = flag_destino_uci OR flag_uci_no_planeada OR flag_complicaciones_medicas
       OR flag_intubacion_dificil OR flag_aferesis OR flag_perdidas_altas
       OR flag_urgencias_30_dias OR flag_hospitalizacion_no_anticipada
```

**Diferencia con `target_d_v2_hosp`:** Esta versión intenta capturar solo los eventos de mayor severidad:
- Incluye `flag_urgencias_30_dias` (visitas a urgencias posoperatorias) y `flag_aferesis` (procedimiento hematológico grave)
- Excluye `flag_seguimiento`, `flag_liquidos`, `flag_tipo_intubacion_complejo` (eventos de severidad moderada)
- Excluye `flag_estancia_uci` (redundante con `flag_uci_no_planeada`)

**Rendimiento:** AUC ~0.76, similar a `target_d_v2_hosp`. Las métricas entre ambas versiones son muy cercanas, lo que indica que la diferencia en composición del target no se traduce en diferencias grandes de rendimiento del modelo.

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

---

## 4. Regla de exclusión de cancelaciones no-médicas

Una regla especial se aplica en todas las versiones del target:

```python
if apply_cancel_non_medico_rule:
    mask = (df["canceladas"] == 1) & (df["canceladas_por_medico"] == 0)
    df.loc[mask, "target"] = 0
```

**¿Por qué?** Un procedimiento cancelado por razones no-médicas (el paciente no llegó, cancelación administrativa, falta de sala) no indica que el paciente necesite valoración preanestésica — la cancelación no fue por un hallazgo clínico. Si se mantienen estos casos como positivos, el modelo aprendería a predecir cancelaciones administrativas, que no es el objetivo. Solo las cancelaciones por razón médica (hallazgo clínico que contraindica proceder) son relevantes para el target.

---

## 5. Distribución del target en la versión seleccionada

Para `target_d_v2_hosp`:

```
Total pacientes en merged: 23,387
├── Negativos (target=0): 16,912 (72.3%)
└── Positivos (target=1):  6,475 (27.7%)

División train/test (80/20 estratificado):
├── Train: 18,709 | 5,180 positivos (27.7%)
└── Test:   4,678 | 1,295 positivos (27.7%)
```

La estratificación garantiza que el desbalance de clases (72.3% / 27.7%) sea idéntico en train y test, evitando que una partición aleatoria desfavorable sesgie la evaluación.

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

Los tres enfoques del [análisis posoperatorio](../analisis-posoperatorio/README.md) aportan evidencia sobre cómo mejorar el target:

1. **Enfoque C** muestra que `flag_hospitalizacion_no_anticipada` (AUC 0.814), `flag_interconsultas` (0.836) y `flag_glucometria_anormal` (0.775) son los flags más predecibles individualmente. El target actual incluye el primero pero no los otros dos.

2. **Enfoque A** (clustering) muestra que los pacientes del Cluster C2 (definidos por `flag_intubado_salida`) tienen perfiles preop casi idénticos a los negativos — confirma que `flag_intubado_salida` no debe estar en el target (excepto si coincide con UCI no planificada).

3. **La propuesta de target redefinido** basada en estos análisis sería:
   ```
   target_propuesto = flag_hospitalizacion_no_anticipada
                   OR flag_uci_no_planeada
                   OR flag_estancia_uci
                   OR flag_interconsultas
                   OR flag_glucometria_anormal
                   OR flag_estancia_prolongada
   ```
   Todos con AUC individual ≥ 0.76 en el Enfoque C, representando eventos predecibles y de alto impacto clínico.
