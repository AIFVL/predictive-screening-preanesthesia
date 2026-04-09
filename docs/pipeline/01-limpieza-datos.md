# Etapa 1 — Limpieza y Preprocesamiento de Datos

**Código fuente:**
- [`src/cleaning/cleaner.py`](../../src/cleaning/cleaner.py) — Pipeline principal de limpieza determinista
- [`src/cleaning/enrichment.py`](../../src/cleaning/enrichment.py) — Enriquecimiento con APIs externas (clasificación de diagnósticos)
- [`src/cleaning/encoding_utils.py`](../../src/cleaning/encoding_utils.py) — Codificación de variables categóricas
- [`src/cleaning/numeric_utils.py`](../../src/cleaning/numeric_utils.py) — Parsing y validación de valores numéricos
- [`src/cleaning/text_utils.py`](../../src/cleaning/text_utils.py) — Normalización de texto libre
- [`src/cleaning/anomaly.py`](../../src/cleaning/anomaly.py) — Detección de anomalías
- [`src/data/loader.py`](../../src/data/loader.py) — Carga de datos brutos

**Outputs:**
- [`output/v1/data_processed/preop_raw.parquet`](../../output/v1/data_processed/preop_raw.parquet) — Dataset preoperatorio limpio y codificado
- [`output/v1/data_processed/posop_raw.parquet`](../../output/v1/data_processed/posop_raw.parquet) — Dataset posoperatorio con flags calculados
- [`output/v1/data_processed/cleaned.parquet`](../../output/v1/data_processed/cleaned.parquet) — Dataset limpio final (mayores de 18 años)
- [`output/v1/reports/cleaning_report.json`](../../output/v1/reports/cleaning_report.json) — Reporte de limpieza
- [`output/v1/reports/validation_report.json`](../../output/v1/reports/validation_report.json) — Validación del dataset

---

## 1. Contexto y desafíos

Los datos brutos del sistema OPERA son registros hospitalarios diseñados para uso clínico, no para modelado estadístico. Esto implica múltiples problemas típicos de datos de salud reales:

- **Texto libre no estandarizado:** campos como diagnósticos, procedimientos y antecedentes médicos son strings libres donde el mismo concepto aparece escrito de decenas de formas distintas ("hta", "HTA", "hipertensión arterial", "hiper arterial", etc.).
- **Valores numéricos con formato inconsistente:** resultados de laboratorio como PT/INR aparecen como "14.5 / 1.2", "1.2 INR", "14seg 1.2", etc.
- **Variables multilabel:** un paciente puede tener múltiples comorbilidades, múltiples alergias, múltiples antecedentes quirúrgicos — todo registrado en un solo campo de texto.
- **Valores fuera de rango biológico:** por errores de digitación o formato, hay edades de 0 o 200, hemoglobinas de 0.1, tensiones arteriales de 999.
- **Pacientes pediátricos:** el sistema registra pacientes de todas las edades, pero el proyecto solo aplica a adultos (≥18 años).
- **Fuga de datos potencial:** algunas columnas del dataset posoperatorio registran información solo conocible después de la cirugía — se deben excluir cuidadosamente de las features.
- **Codificación de caracteres:** nombres de columnas con caracteres especiales (ñ, tildes) presentaron problemas de encoding (UTF-8 vs. Latin-1) que generaron columnas con nombres como "TensiÃ³n Arterial".

---

## 2. Dataset preoperatorio — Flujo de limpieza

### 2.1 Dimensiones iniciales y finales

| Etapa | Filas | Columnas |
|-------|-------|----------|
| Datos brutos | 30,962 | 236 |
| Tras filtrado de edad (≥18) | 24,279 | 236 |
| Tras limpieza y codificación | 24,279 | 238 (+2 columnas enriquecidas) |

El filtro de edad elimina **6,683 registros** (21.6% del total) correspondientes a pacientes pediátricos — una porción significativa que confirma que el sistema incluye cirugías pediátricas.

### 2.2 Variables numéricas — Validación y parseo

Las variables numéricas del dataset presentan tres problemas comunes: texto incrustado en el valor, separadores decimales inconsistentes (punto vs. coma), y valores biológicamente imposibles por error de digitación.

**Parseo robusto de resultados de laboratorio (`src/cleaning/numeric_utils.py`):**

```python
def _to_float(s, default=None):
    # Maneja: "14.5", "14,5", "~14.5", "<14.5", ">14.5", "≈14.5"
    raw = str(s).strip()
    tmp = raw.replace(",", ".")
    tmp = re.sub(r"[<>≈~]", "", tmp)
    try:
        return float(tmp)
    except:
        # Extrae primer número válido del string
        m = re.search(r"[-+]?\d*\.?\d+", tmp)
        return float(m.group()) if m else default
```

**Caso especial — PT/INR:** El tiempo de protrombina (PT) aparece registrado junto al INR en formatos extremadamente variables: "14.5 / 1.2", "1.2 INR", "14seg normal 1.2", "14.5/11.5 control". El parser `_pt_inr_from_text()` extrae específicamente el INR:
1. Extrae todos los números del string.
2. Si hay dos números, identifica cuál es PT (7–1000 segundos) y cuál es INR (0.7–8).
3. Si solo hay un número, determina si es PT o INR por el rango.
4. Si es PT, convierte a INR usando la fórmula `INR = (PT / PT_control)^ISI`.

**Reglas de outliers (configuradas en `config/cleaning_config.yaml`):**
```yaml
outlier_rules:
  Edad:
    min: 0
    max: 120
```
Valores fuera de rango se convierten a `NaN` y luego se imputan.

### 2.3 Variables categóricas — Normalización de texto libre

El módulo `src/cleaning/text_utils.py` implementa un pipeline de normalización de texto:

```
texto bruto → minúsculas → eliminar acentos → eliminar caracteres especiales
→ tokenización → normalización de tokens → resolución de contradicciones
→ lista de términos estandarizados
```

**Función `normalize_tokens()`:** Mapea variantes ortográficas al término canónico. Por ejemplo:
- "hta", "hipertension arterial", "hiper", "ht" → `hta`
- "epoc", "enfermedad pulmonar obstructiva", "epco" → `epoc`
- "diabetes", "dm", "dm2", "diabetes mellitus" → `diabetes`

**Función `resolve_contradictory_term()`:** Maneja el caso de registros que dicen "negativo" para un sistema pero luego listan comorbilidades positivas en el mismo campo — la regla es que los términos positivos específicos tienen precedencia sobre "negativo".

### 2.4 Variables multilabel — Codificación

Las variables de antecedentes médicos (cardiovasculares, respiratorios, neurológicos, etc.) son campos de texto libre que pueden contener múltiples condiciones. Se procesan con `encode_multilabel()`:

**Proceso:**
1. El texto libre se normaliza con `normalize_tokens()`.
2. Se genera un conjunto de términos para ese registro.
3. Para cada término conocido en el vocabulario del sistema, se crea una columna binaria separada.

**Resultado:** Un campo como "Antecedentes cardiovasculares" que contenía strings variables como "hta, arritmia, marcapaso" se convierte en columnas binarias independientes:

| `Antecedentes cardiovasculares_hta` | `Antecedentes cardiovasculares_arritmias` | `Antecedentes cardiovasculares_cardiopatias` | ... |
|-----|------|------|-----|
| 1 | 1 | 0 | ... |

Este patrón se aplica a: antecedentes cardiovasculares, respiratorios, neurológicos, hematológicos, endocrinológicos, renales, gastrointestinales, anestesia previa, tipo de anestesia propuesta, condición actual, alergias, etc.

### 2.5 Variables de diagnóstico y procedimiento — Clasificación con LLM

Las columnas `Dx Preoperatorio`, `Procedimiento propuesto` y `Antecedentes quirúrgicos` contienen descripciones en texto libre de diagnósticos y procedimientos médicos. La variabilidad es enorme: el mismo diagnóstico puede estar escrito como "cancer de colon", "Adenocarcinoma de colon", "neo colon", "Ca. colon" o simplemente un código CIE-10 incompleto.

**Solución implementada (`src/cleaning/enrichment.py`):**

Se utiliza un **modelo de lenguaje (LLM)** para clasificar cada descripción de diagnóstico o procedimiento en el capítulo del código CIE-10 o clasificación de procedimiento correspondiente. El resultado se codifica como variables One-Hot:

```
"cancer de colon" → Capítulo C (Neoplasias malignas) → Dx_Preoperatorio_Code_C = 1
"fractura de fémur" → Capítulo S (Traumatismos) → Dx_Preoperatorio_Code_S = 1
```

Esto genera columnas como `Dx Preoperatorio Code_A`, `Dx Preoperatorio Code_B`, ..., `Dx Preoperatorio Code_Z`, `Dx Preoperatorio Code_NO ENCONTRADO`, una por cada capítulo CIE-10 presente en los datos.

El mismo proceso se aplica a procedimientos, generando `Procedimiento propuesto Code_*` y `Antecedentes quirúrgicos Code_*`.

### 2.6 Scores de severidad — Ingeniería de features clínica

Uno de los aportes más importantes de la limpieza es la generación de **scores de severidad** para diagnósticos y procedimientos, una transformación que convierte texto clínico en información cuantitativa estructurada.

Para cada diagnóstico y procedimiento, el enriquecimiento con LLM también asigna una categoría de severidad clínica. A partir de estas categorías se calculan scores binarios que indican si el diagnóstico/procedimiento del paciente es de alta, media, baja severidad, etc.:

| Feature | Descripción |
|---------|-------------|
| `score_proc_high_severity` | Procedimiento de alta severidad (1/0) |
| `score_proc_critical` | Procedimiento crítico (1/0) |
| `score_proc_moderate_severity` | Procedimiento de severidad moderada (1/0) |
| `score_proc_medium_severity` | Procedimiento de severidad media (1/0) |
| `score_proc_low_severity` | Procedimiento de baja severidad (1/0) |
| `score_dx_high_severity` | Diagnóstico de alta severidad (1/0) |
| `score_dx_critical` | Diagnóstico crítico (1/0) |
| `score_dx_moderate_severity` | Diagnóstico de severidad moderada (1/0) |
| `score_dx_medium_severity` | Diagnóstico de severidad media (1/0) |
| `score_dx_low_severity` | Diagnóstico de baja severidad (1/0) |
| `score_proc_ant_*` | Mismo esquema para antecedentes quirúrgicos |

Estos scores son las features **más importantes** del modelo final, como veremos en la etapa de selección de features.

### 2.7 Variables de tiempo

De las fechas de la valoración preanestésica se extraen:
- `anio` — año del registro
- `mes` — mes del registro (1–12)
- `dia_semana` — día de la semana (0=lunes, 6=domingo)
- `Hora_decimal` — hora del día como número decimal (e.g., 14:30 → 14.5)

Estas variables capturan potenciales efectos temporales: puede haber diferencias en la complejidad de los casos entre días de semana y fines de semana, o entre meses del año.

### 2.8 Corrección de encoding de columnas

Los nombres de algunas columnas llegaron con errores de encoding (probablemente exportadas de un sistema con codificación Latin-1 y leídas como UTF-8):

```python
encoding_fix_map = {
    "TensiÃ³n Arterial SistÃ³lica (mm/Hg)": "Tensión Arterial Sistólica (mm/Hg)",
    "PrÃ³tesis Dental_movil": "Prótesis Dental_movil",
    "AlÃ©rgeno_med_opioides": "Alérgeno_med_opioides",
    # ... más correcciones
}
```

### 2.9 Imputación de valores faltantes

Después de la limpieza, el dataset preoperatorio tiene **0% de valores nulos** (`null_pct_after: 0.0` en `cleaning_report.json`). La imputación se realiza:
- **Variables numéricas:** mediana de la columna (robusta a outliers).
- **Variables categóricas binarias:** 0 (ausencia del antecedente/condición).
- **Valores fuera de rango biológico:** se reemplazan por el valor límite del rango o por la mediana.

---

## 3. Dataset posoperatorio — Construcción de flags

El dataset posoperatorio (`posop_raw.parquet`) es procesado de manera diferente: en lugar de limpiar variables para modelado directo, el objetivo es **construir flags binarios** que resumen aspectos clínicamente relevantes de lo que ocurrió durante y después de la cirugía.

### 3.1 Estructura de flags

El pipeline de flags (`src/target/pipeline.py` con módulos en `src/target/specific/`) produce **57 flags** organizados en categorías:

**Flags de reservas de sangre:**
- `flag_reserva_hemoderivados` — se reservaron hemoderivados para el procedimiento
- `flag_reserva_sangre` — se reservó sangre para el procedimiento
- `flag_reservas` — superset: cualquier reserva (sangre o hemoderivados)

**Flags fisiológicos (signos vitales intraoperatorios):**
- `flag_presion_sistolica_anormal` — tensión arterial sistólica prequirúrgica fuera de rango normal
- `flag_presion_diastolica_anormal` — tensión diastólica anormal
- `flag_presion_media_anormal` — presión media anormal
- `flag_frecuencia_cardiaca_anormal` — frecuencia cardíaca anormal
- `flag_saturacion_oxigeno_anormal` — saturación de oxígeno anormal
- `flag_temperatura_anormal` — temperatura anormal
- `flag_glucometria_anormal` — glucometría anormal
- `flag_fisiologicas` — superset: cualquier alteración fisiológica

**Flags de tiempos quirúrgicos:**
- `flag_duracion_cirujano_larga` — duración quirúrgica mayor al percentil 90 para el tipo de procedimiento
- `flag_duracion_anestesia_larga` — duración anestésica mayor al percentil 90
- `flag_tiempos` — superset: cualquier tiempo largo

**Flags de vía aérea:**
- `flag_induccion_compleja` — inducción anestésica compleja (secuencia de inducción rápida u otro protocolo especial)
- `flag_intubacion_dificil` — intubación difícil (>2 intentos, uso de dispositivo alternativo)
- `flag_laringoscopia_alta` — laringoscopia Cormack-Lehane III o IV
- `flag_tipo_intubacion_complejo` — tipo de intubación no estándar
- `flag_hoja_laringoscopio_recta` — uso de hoja recta (Miller) en lugar de la curva estándar
- `flag_elemento_via_aerea_complejo` — uso de dispositivo complejo (videolaringoscopio, fibroscopio)
- `flag_via_aerea` — superset: cualquier evento de vía aérea

**Flags de ventilación:**
- `flag_ventilacion_asistida` — ventilación asistida en algún momento
- `flag_control_manual` — ventilación en modo control manual
- `flag_modos_avanzados` — modos ventilatorios avanzados (SIMV, presión soporte)
- `flag_parametros_ventilatorios` — parámetros ventilatorios anormales
- `flag_frecuencia_resp_anormal` — frecuencia respiratoria anormal
- `flag_ventilacion` — superset: cualquier evento ventilatorio

**Flags de técnica anestésica:**
- `flag_tecnica_combinada` — técnica anestésica combinada (general + regional)
- `flag_monitoreo_invasivo` — monitoreo invasivo (catéter arterial, PVC)
- `flag_tecnica` — superset: técnica compleja

**Flags de líquidos:**
- `flag_hemoderivados` — se administraron hemoderivados
- `flag_aferesis` — procedimiento de aféresis
- `flag_volumen_alto` — volumen de líquidos administrado alto
- `flag_perdidas_altas` — volumen de líquidos eliminados alto
- `flag_balance_extremo` — balance hídrico extremo (positivo o negativo marcado)
- `flag_liquidos` — superset: cualquier evento de líquidos

**Flags de desenlace inmediato:**
- `flag_destino_uci` — destino final de quirófano: UCI
- `flag_intubado_salida` — sale intubado de quirófano
- `flag_no_despierto` — sale sin despertar de la anestesia

**Flags de complicaciones clínicas:**
- `flag_complicacion` — superset: cualquier complicación registrada
- `flag_infarto_miocardio` — infarto de miocardio perioperatorio
- `flag_acv_hemorragico` — accidente cerebrovascular hemorrágico
- `flag_tia` — ataque isquémico transitorio
- `flag_aspiracion_pulmonar` — aspiración pulmonar
- `flag_complicaciones_pulmonares` — complicaciones pulmonares posoperatorias
- `flag_complicaciones_medicas` — complicaciones médicas generales

**Flags de estancia:**
- `flag_estancia_prolongada` — estancia hospitalaria > 3 días de lo esperado para el procedimiento
- `flag_hospitalizacion_no_anticipada` — hospitalización que no estaba planificada
- `flag_uci_no_planeada` — ingreso a UCI no planificado
- `flag_estancia_uci` — estancia en UCI (planeada o no)
- `flag_estancia` — superset: cualquier evento de estancia

**Flags de seguimiento posoperatorio:**
- `flag_urgencias_30_dias` — consulta a urgencias en los 30 días siguientes
- `flag_interconsultas` — necesidad de interconsulta especializada
- `flag_seguimiento` — superset: cualquier seguimiento adicional

**Flag de cancelación:**
- `flag_cancelacion` — procedimiento cancelado
- El target excluye automáticamente cancelaciones no-médicas (cancelaciones por razones administrativas o de paciente, no médicas)

### 3.2 Lógica de cada flag

Cada flag tiene una función dedicada en `src/target/specific/`. Por ejemplo:

**`flag_intubacion_dificil`** se calcula en `via_aerea.py`:
- El campo `Intubación` del registro anestésico contiene categorías como "intubación normal", "IOT difícil", "fibroscopia", etc.
- La lógica clasifica como `flag_intubacion_dificil = 1` cuando el registro indica cualquier variante de intubación difícil o de rescate.

**`flag_duracion_cirujano_larga`** se calcula en `tiempos.py`:
- Se calcula la duración en minutos entre `Inicio cirujano` y `Fin cirujano`.
- Se compara con un percentil de referencia apropiado para el tipo de procedimiento.
- Si la duración supera el umbral, `flag_duracion_cirujano_larga = 1`.

**`flag_hospitalizacion_no_anticipada`** viene directamente del campo de texto `Hospitalización no anticipada` del registro, convertido a binario.

---

## 4. Validación del dataset

El reporte de validación (`output/v1/reports/validation_report.json`) confirma:

| Métrica | Preoperatorio | Posoperatorio |
|---------|---------------|---------------|
| Filas | 30,962 | 29,865 |
| Columnas | 236 | 134 |
| % nulos | 0.0% | 5.32% |

**¿Por qué el posoperatorio tiene 5.32% de nulos?** Algunos campos de la hoja anestésica no se completan para todos los procedimientos (por ejemplo, los parámetros ventilatorios no aplican para cirugías con anestesia local o sedación mínima). Estos valores faltantes se manejan durante la construcción de flags con reglas específicas por campo.

---

## 5. Reporte de limpieza

El archivo `output/v1/reports/cleaning_report.json` resume:

```json
{
    "rows_before": 30962,
    "rows_after": 24279,
    "rows_removed": 6683,
    "cols_before": 236,
    "cols_after": 238,
    "null_pct_before": 0.0,
    "null_pct_after": 0.0
}
```

**Notas:**
- `cols_after > cols_before`: Se añadieron 2 columnas de scores de severidad durante el enriquecimiento.
- `null_pct_after = 0.0`: La imputación elimina todos los valores faltantes del dataset preoperatorio.
- `rows_removed = 6683`: Solo por filtro de edad (menores de 18 años). No se eliminaron filas por calidad de datos.

---

## 6. Análisis exploratorio (EDA)

Los EDA están disponibles como gráficos en `output/v1/plots/`:

- `output/v1/plots/eda_preop_raw/` — distribuciones de variables del dataset preoperatorio antes de la limpieza
- `output/v1/plots/eda_preop_clean/` — distribuciones después de la limpieza
- `output/v1/plots/eda_posop_target_d_v2_hosp/` — distribuciones de flags posoperatorios para el target seleccionado

Los análisis de correlación pre-post (qué variables preoperatorias se correlacionan con el target) están en `output/v1/reports/pre_post_signal/` — ver detalles en [03-seleccion-features.md](03-seleccion-features.md).
