# Enfoque A — Estructuras Internas del Dataset Posoperatorio

> **Pregunta central:** ¿Existen tipos naturales de "paciente complicado" en el dataset posoperatorio? ¿Sus perfiles preoperatorios difieren entre grupos?

**Código fuente:** [`src/analysis/posop_clustering.py`](../../src/analysis/posop_clustering.py)
**Output CSV (labels):** [`output/v1/reports/analisis_posoperatorio/clustering_labels.csv`](../../output/v1/reports/analisis_posoperatorio/clustering_labels.csv)
**Output CSV (perfil):** [`output/v1/reports/analisis_posoperatorio/clustering_profile.csv`](../../output/v1/reports/analisis_posoperatorio/clustering_profile.csv)
**Output PNG:** [`output/v1/reports/analisis_posoperatorio/posop_clustering.png`](../../output/v1/reports/analisis_posoperatorio/posop_clustering.png)

---

## 1. Motivación

El target actual del modelo es una variable binaria (0/1): ¿el paciente presentó alguna complicación o evento posoperatorio? Pero "alguna complicación" puede significar cosas muy distintas. Un paciente con UCI no planificada y otro que salió intubado como decisión rutinaria del anestesiólogo son ambos "positivos" en el target, pero sus situaciones clínicas son radicalmente diferentes.

Este análisis aplica clustering no supervisado directamente sobre los patrones de flags posoperatorios para responder: ¿si dejamos que los datos hablen solos, en qué grupos naturales se agrupan los pacientes posoperatorios? ¿Tienen esos grupos perfiles preoperatorios distinguibles?

Un hallazgo de clusters con perfiles clínicos distintos y predecibles desde preop sería evidencia de que el problema no es que "las variables preop no predicen complicaciones", sino que hay múltiples tipos de complicación y el modelo actual intenta aprender todos con una sola señal.

---

## 2. Metodología

### 2.1 Datos de entrada

Se utilizan dos fuentes:

- **`posop_raw.parquet`** (29,865 registros): contiene los 18 flags clínicos seleccionados para clustering (ver lista en sección 2.2).
- **`merged.parquet`** (`target_d_v2_hosp`, 23,387 registros): contiene variables preoperatorias para enriquecer el perfil de cada cluster.

El join entre ambas fuentes se realiza por la clave `Documento PMD (valoración preanestésica)` ↔ `Documento PMD`, resultando en **23,387 pacientes con datos completos** para el análisis.

### 2.2 Flags seleccionados para clustering

Se seleccionan 18 flags que representan **complicaciones y eventos clínicos de resultado**, excluyendo flags de técnica, proceso o decisiones intraoperatorias:

| Flag | ¿Qué representa? |
|------|-----------------|
| `flag_intubacion_dificil` | Intubación difícil durante el procedimiento |
| `flag_tipo_intubacion_complejo` | Tipo de intubación no estándar o complejo |
| `flag_elemento_via_aerea_complejo` | Uso de dispositivos complejos para vía aérea |
| `flag_aferesis` | Necesidad de aféresis (procedimiento hematológico) |
| `flag_balance_extremo` | Balance hídrico extremo durante la cirugía |
| `flag_liquidos` | Volumen de líquidos administrado elevado |
| `flag_destino_uci` | Destino posquirúrgico: UCI |
| `flag_intubado_salida` | Paciente sale de quirófano intubado |
| `flag_complicaciones_pulmonares` | Complicaciones pulmonares posoperatorias |
| `flag_complicaciones_medicas` | Complicaciones médicas generales posoperatorias |
| `flag_estancia_prolongada` | Estancia hospitalaria prolongada |
| `flag_hospitalizacion_no_anticipada` | Hospitalización no planificada |
| `flag_uci_no_planeada` | Ingreso a UCI no planificado |
| `flag_estancia_uci` | Estancia en UCI |
| `flag_urgencias_30_dias` | Consulta a urgencias en 30 días postop |
| `flag_interconsultas` | Necesidad de interconsulta especializada |
| `flag_reserva_sangre` | Se reservó sangre para el procedimiento |
| `flag_reserva_hemoderivados` | Se reservaron hemoderivados |

Los valores faltantes en estos flags se imputan con 0 (ausencia del evento).

### 2.3 Algoritmo: KMeans

Se aplica **KMeans** con los siguientes parámetros:

```
KMeans(
    n_clusters=5,
    random_state=42,
    n_init=20,      # 20 inicializaciones aleatorias para evitar mínimos locales
)
```

**¿Por qué 5 clusters?** Se eligió 5 como número de clusters para capturar variedad clínica sin excesiva fragmentación. En clustering médico, un número entre 4 y 6 suele balancear interpretabilidad y granularidad. En una iteración futura se podría usar el método del codo o silueta para validar el número óptimo.

**¿Por qué KMeans sobre datos binarios?** KMeans usa distancia euclidiana, que sobre datos binarios equivale a la distancia de Hamming (número de diferencias entre dos vectores binarios). Es una aproximación válida para este propósito exploratorio. Alternativas como K-Modes (diseñadas específicamente para datos categóricos) podrían usarse en una validación más rigurosa.

Los datos se usan directamente sin normalización, ya que todos los flags ya están en escala 0/1.

### 2.4 Visualización: PCA 2D

Para visualizar los clusters en un plano 2D, se aplica **Análisis de Componentes Principales (PCA)** con 2 componentes, previa estandarización (`StandardScaler`).

La estandarización antes del PCA es importante: aunque los flags son binarios, tienen distintas prevalencias. Sin estandarizar, los flags más frecuentes dominarían el espacio PCA.

**Varianza explicada por PCA:**
- PC1: 22.7% de la varianza total
- PC2: 10.9% de la varianza total
- **Total: 33.6%**

El 33.6% de varianza explicada en 2D es razonable para datos binarios de alta dimensionalidad. No significa que el clustering sea impreciso — el clustering se hizo en el espacio de 18 dimensiones; el PCA solo sirve para visualización.

### 2.5 Perfil preoperatorio por cluster

Para cada cluster se calculan los promedios de 11 variables preoperatorias seleccionadas:

| Variable preop | Descripción |
|----------------|-------------|
| `Edad` | Edad del paciente |
| `IMC` | Índice de masa corporal |
| `Tensión Arterial Sistólica (mm/Hg)` | Tensión sistólica basal |
| `score_proc_high_severity` | Score de alta severidad del procedimiento |
| `score_proc_moderate_severity` | Score de severidad moderada del procedimiento |
| `score_dx_high_severity` | Score de alta severidad del diagnóstico |
| `Puntaje Mallampati` | Clasificación de vía aérea |
| `Antecedentes cardiovasculares_hta` | Hipertensión arterial (binario) |
| `Antecedente endocrinológicos_diabetes` | Diabetes (binario) |
| `Antecedente respiratorios_epoc` | EPOC (binario) |
| `Antecedente respiratorios_asma` | Asma (binario) |

---

## 3. Resultados

### 3.1 Visión general de los clusters

| Cluster | N pacientes | % del total | Target (complicó) | Perfil dominante |
|---------|-------------|-------------|-------------------|-----------------|
| C0 | 21,489 | **91.9%** | 42.3% | Grupo mayoritario — bajo/mixto riesgo |
| C1 | 671 | 2.9% | **100%** | Complicaciones graves con UCI + balance extremo |
| C2 | 3,008 | 12.9% | **100%** | Intubado al salir de quirófano |
| C3 | 3,979 | 17.0% | **100%** | Hospitalización no anticipada + estancia prolongada |
| C4 | 718 | 3.1% | **100%** | Aféresis + líquidos |

> **Nota importante:** Los clusters 1–4 tienen target_prevalence = 1.0, lo que significa que TODOS los pacientes en esos clusters tienen el target positivo. Solo el cluster C0 tiene mezcla de positivos (42.3%) y negativos. Esto revela que el clustering está capturando grupos homogéneos en términos de complejidad posoperatoria.

---

### 3.2 Análisis cluster por cluster

#### Cluster C0 — El grupo mayoritario (n=21,489, 91.9%)

Este es el cluster "residual" — captura la gran mayoría de los pacientes, incluyendo tanto pacientes sin complicaciones como pacientes con complicaciones leves o de tipo mixto que no encajan en ninguno de los fenotipos específicos de los otros clusters.

**Perfil de flags (prevalencias):**

| Flag | Prevalencia en C0 | Observación |
|------|-------------------|-------------|
| `flag_urgencias_30_dias` | 3.12% | La complicación más frecuente en este grupo |
| `flag_tipo_intubacion_complejo` | 2.97% | Presente en pequeña proporción |
| `flag_intubacion_dificil` | 1.50% | Baja prevalencia |
| Todos los demás flags | 0.00% – 0.10% | Prácticamente ausentes |

Los flags de alta gravedad (UCI, hospitalización no anticipada, estancia prolongada) son prácticamente 0 en este cluster. El 42.3% de target positivo dentro de C0 probablemente corresponde a pacientes con complicaciones muy leves o eventos de baja importancia clínica.

**Perfil preoperatorio:**

| Variable | C0 |
|----------|-----|
| Edad media | 46.2 años |
| IMC medio | 25.8 |
| TA sistólica media | 119.8 mmHg |
| Score proc. alta severidad | 0.296 |
| Score dx. alta severidad | 0.302 |
| Mallampati medio | 0.476 |
| % con HTA | 8.5% |
| % con diabetes | 3.0% |
| % con EPOC | 0.4% |
| % con asma | 2.2% |

**Interpretación:** El perfil de C0 es el del paciente "típico" de la clínica: adulto de mediana edad, IMC normal, tensión normal, pocas comorbilidades. No hay una señal de riesgo alta en su perfil preop. El 42.3% de target positivo dentro de este cluster es esperable dado que el target compuesto actual incluye eventos muy leves que cualquier paciente puede presentar.

---

#### Cluster C1 — Complicaciones graves con requerimiento de UCI (n=671, 2.9%)

**Perfil de flags:**

| Flag | Prevalencia en C1 | vs. C0 |
|------|-------------------|--------|
| `flag_hospitalizacion_no_anticipada` | **100.0%** | +100% |
| `flag_uci_no_planeada` | **99.9%** | +100% |
| `flag_estancia_uci` | **99.9%** | +100% |
| `flag_estancia_prolongada` | **72.3%** | +72% |
| `flag_complicaciones_medicas` | 16.2% | +16% |
| `flag_complicaciones_pulmonares` | 14.3% | +14% |
| `flag_balance_extremo` | 13.3% | +13% |
| `flag_liquidos` | 15.8% | +16% |
| `flag_intubado_salida` | 27.9% | +28% |
| `flag_destino_uci` | 49.8% | +50% |
| `flag_interconsultas` | 31.5% | +31% |
| `flag_reserva_hemoderivados` | 25.2% | +25% |
| `flag_reserva_sangre` | 18.9% | +19% |

Este cluster es el de las **complicaciones más graves y multisistémicas**. Todos los pacientes tuvieron hospitalización no anticipada, prácticamente todos requirieron UCI, la mayoría tuvo estancia prolongada. Además hay balance hídrico extremo, complicaciones médicas y pulmonares — un cuadro de alta complejidad posoperatoria.

**Perfil preoperatorio:**

| Variable | C1 | C0 | Diferencia |
|----------|----|----|-----------|
| Edad media | 48.1 años | 46.2 | +2 años |
| IMC medio | 25.7 | 25.8 | ≈ igual |
| TA sistólica media | 121.7 mmHg | 119.8 | +2 mmHg |
| Score proc. alta severidad | 0.268 | 0.296 | **-10%** (¡menor!) |
| Score dx. alta severidad | 0.315 | 0.302 | +4% |
| Mallampati medio | 0.607 | 0.476 | **+28%** |
| % con HTA | **19.5%** | 8.5% | **+130%** |
| % con diabetes | **8.3%** | 3.0% | **+177%** |
| % con EPOC | **1.1%** | 0.4% | +175% |

**Interpretación:** Esto es muy revelador. Los pacientes del Cluster C1 (los más graves posoperatoriamente) tienen:
- **Más HTA (19.5% vs. 8.5%)** — casi el doble
- **Más diabetes (8.3% vs. 3.0%)** — casi el triple
- **Mayor Mallampati** — más vía aérea difícil
- **PARADÓJICAMENTE: score de procedimiento MENOR** — las complicaciones graves no siempre vienen de los procedimientos más complejos

Esto sugiere que las **comorbilidades del paciente** (HTA, diabetes) son más determinantes para las complicaciones graves que la complejidad del procedimiento. Es el hallazgo más importante del clustering para la redefinición del modelo.

---

#### Cluster C2 — Pacientes intubados al salir de quirófano (n=3,008, 12.9%)

**Perfil de flags:**

| Flag | Prevalencia en C2 | vs. C0 |
|------|-------------------|--------|
| `flag_intubado_salida` | **100.0%** | +100% |
| `flag_hospitalizacion_no_anticipada` | 16.1% | +16% |
| `flag_urgencias_30_dias` | 4.0% | ≈ igual |

Este cluster está **completamente definido** por `flag_intubado_salida`. Un solo flag domina la pertenencia al cluster. El 16.1% de hospitalización no anticipada muestra que algunos de estos pacientes tuvieron consecuencias adicionales, pero la mayoría se intubó por decisión rutinaria del anestesiólogo (procedimientos que normalmente terminan con el paciente intubado en UCI, como cirugías cardíacas o de tórax).

**Perfil preoperatorio:**

| Variable | C2 | C0 | Diferencia |
|----------|----|----|-----------|
| Edad media | 45.0 años | 46.2 | ≈ igual |
| IMC medio | 25.9 | 25.8 | ≈ igual |
| Score proc. alta severidad | 0.303 | 0.296 | +2% |
| % con HTA | 8.2% | 8.5% | ≈ igual |
| % con diabetes | 2.8% | 3.0% | ≈ igual |

**Interpretación:** El perfil preoperatorio de C2 es **casi idéntico al de C0**. Los pacientes que salen intubados no tienen un perfil preoperatorio sustancialmente diferente al grupo general. Esto confirma lo que el Enfoque C ya mostró: `flag_intubado_salida` tiene AUC 0.55 — prácticamente aleatorio desde variables preop. La decisión de salir intubado es principalmente intraoperatoria.

**Implicación para el target:** Si C2 está completamente definido por un flag no predecible desde preop, su inclusión en el target compuesto solo añade ruido al modelo. 3,008 pacientes (12.9% del total) están contaminando la señal del target.

---

#### Cluster C3 — Hospitalización no anticipada y estancia prolongada (n=3,979, 17.0%)

**Perfil de flags:**

| Flag | Prevalencia en C3 | vs. C0 |
|------|-------------------|--------|
| `flag_hospitalizacion_no_anticipada` | **100.0%** | +100% |
| `flag_estancia_prolongada` | 14.5% | +14% |
| `flag_interconsultas` | 24.5% | +24% |
| `flag_urgencias_30_dias` | 8.2% | +5% |
| `flag_balance_extremo` | 3.5% | +4% |
| `flag_intubado_salida` | 3.4% | +3% |

El driver de este cluster es `flag_hospitalizacion_no_anticipada` (100%). Es el cluster más numeroso de los "graves" (n=3,979). A diferencia de C1 (que también tiene hospitalización no anticipada + UCI), este cluster se hospitaliza de forma no esperada pero en su mayoría no requiere UCI.

**Perfil preoperatorio:**

| Variable | C3 | C0 | Diferencia |
|----------|----|----|-----------|
| Edad media | 46.5 años | 46.2 | ≈ igual |
| IMC medio | 26.7 | 25.8 | **+3.5%** |
| TA sistólica media | 121.5 mmHg | 119.8 | +1.4% |
| Score proc. alta severidad | **0.327** | 0.296 | **+10%** |
| Score dx. alta severidad | 0.318 | 0.302 | +5% |
| Mallampati medio | 0.533 | 0.476 | **+12%** |
| % con HTA | **10.2%** | 8.5% | +20% |
| % con diabetes | **3.8%** | 3.0% | +27% |

**Interpretación:** C3 tiene un perfil de riesgo moderadamente más elevado que C0: mayor score de severidad del procedimiento (+10%), mayor Mallampati, más HTA y diabetes. Es el "paso intermedio" entre el paciente promedio (C0) y las complicaciones graves (C1). La hospitalización no anticipada en este grupo parece determinada por la complejidad del procedimiento más que por comorbilidades extremas.

**`flag_hospitalizacion_no_anticipada`** tiene AUC 0.814 en el Enfoque C — el tercer flag más predecible. El cluster C3 representa exactamente a los pacientes que el modelo podría detectar si se entrenara específicamente para este evento.

---

#### Cluster C4 — Pacientes con aféresis y uso intensivo de líquidos (n=718, 3.1%)

**Perfil de flags:**

| Flag | Prevalencia en C4 | vs. C0 |
|------|-------------------|--------|
| `flag_aferesis` | **95.8%** | +96% |
| `flag_liquidos` | **100.0%** | +100% |
| `flag_intubado_salida` | 7.2% | +7% |
| `flag_hospitalizacion_no_anticipada` | 19.2% | +19% |
| `flag_urgencias_30_dias` | 3.8% | ≈ igual |
| `flag_balance_extremo` | 4.3% | +4% |

Este cluster es el más especializado: prácticamente definido por **aféresis + volumen alto de líquidos**. Son pacientes hematológicos, probablemente con enfermedades como anemia severa, trombocitopenia, o trastornos de coagulación que requieren transfusiones y aféresis.

**Perfil preoperatorio:**

| Variable | C4 | C0 | Diferencia |
|----------|----|----|-----------|
| Edad media | **43.0 años** | 46.2 | **-7%** — más jóvenes |
| IMC medio | 26.2 | 25.8 | ≈ igual |
| TA sistólica media | 121.4 mmHg | 119.8 | ≈ igual |
| Score proc. alta severidad | 0.327 | 0.296 | +10% |
| Score dx. alta severidad | 0.310 | 0.302 | ≈ igual |
| Mallampati medio | 0.484 | 0.476 | ≈ igual |
| % con HTA | **6.7%** | 8.5% | **-21%** — menos HTA |
| % con diabetes | **2.1%** | 3.0% | -30% — menos diabetes |
| % con EPOC | 0.0% | 0.4% | ≈ 0 |
| % con asma | **1.6%** | 2.2% | menos |

**Interpretación:** Este cluster es clínicamente peculiar: son pacientes más jóvenes, con **menos comorbilidades** (menos HTA, menos diabetes, menos EPOC) que el promedio, pero con procedimientos de mayor complejidad y necesidad de aféresis + líquidos. Esto apunta a una patología hematológica de base (no metabólica ni cardiovascular), donde la complicación posoperatoria es inherente a la enfermedad hematológica subyacente, no a comorbilidades crónicas.

Que `flag_aferesis` tenga AUC 0.595 en el Enfoque C se explica aquí: los pacientes de C4 no tienen el perfil preop "clásico" de riesgo (no son hipertensos ni diabéticos), son jóvenes. El modelo basado en esas features no puede identificarlos fácilmente.

---

### 3.3 Comparación de perfiles preoperatorios entre todos los clusters

| Variable preop | C0 | C1 (UCI grave) | C2 (intubado) | C3 (hosp.) | C4 (aféresis) |
|----------------|----|--------|-------|------|--------|
| Edad | 46.2 | 48.1 | 45.0 | 46.5 | 43.0 |
| IMC | 25.8 | 25.7 | 25.9 | 26.7 | 26.2 |
| TA sistólica | 119.8 | 121.7 | 119.7 | 121.5 | 121.4 |
| Score proc. alta sev. | 0.296 | 0.268 | 0.303 | 0.327 | 0.327 |
| Score dx. alta sev. | 0.302 | 0.315 | 0.308 | 0.318 | 0.310 |
| Mallampati | 0.476 | **0.607** | 0.474 | 0.533 | 0.484 |
| HTA (%) | 8.5% | **19.5%** | 8.2% | 10.2% | 6.7% |
| Diabetes (%) | 3.0% | **8.3%** | 2.8% | 3.8% | 2.1% |
| EPOC (%) | 0.4% | **1.1%** | 0.3% | 0.3% | 0.0% |
| Asma (%) | 2.2% | 2.3% | 2.3% | 2.3% | 1.6% |

---

## 4. Visualización PCA

El scatter plot PCA muestra cómo se distribuyen los 5 clusters en el espacio de 2 componentes principales. Con 33.6% de varianza explicada, la separación visual es parcial pero informativa:

- **C0** (azul, el mayoritario) ocupa el centro y gran parte del espacio — es el cluster más difuso.
- **C1** (complicaciones graves + UCI) tiende a estar en un extremo del PC1, separado del resto.
- **C2** (intubado salida) muestra una banda clara en el espacio PCA — `flag_intubado_salida` domina una dirección específica.
- **C3** (hospitalización no anticipada) parcialmente superpuesto con C0 en el PCA, lo que tiene sentido dado que estos pacientes tienen perfiles preop similares a C0.
- **C4** (aféresis) claramente separado de los demás en el espacio PCA, confirmando su perfil clínico único.

El heatmap de flags por cluster confirma visualmente la interpretación: cada cluster tiene un patrón dominante de flags que lo define.

---

## 5. Implicaciones del clustering

### 5.1 Las complicaciones posoperatorias no son un fenómeno homogéneo

El resultado más importante del clustering es que hay **fenotipos clínicamente distintos** de "paciente complicado":

- **Fenotipo grave-multisistémico (C1):** comorbilidades altas (HTA, diabetes), complicaciones severas con UCI. Predecible desde preop por las comorbilidades.
- **Fenotipo quirúrgico-intubado (C2):** sin diferencias preop vs. el grupo general, la "complicación" es prácticamente una decisión intraoperatoria. No predecible desde preop.
- **Fenotipo hospitalización inesperada (C3):** procedimientos más complejos, hospitalización como desenlace. Predecible desde preop por la complejidad del procedimiento.
- **Fenotipo hematológico (C4):** pacientes más jóvenes, sin comorbilidades clásicas, pero con patología hematológica de base. Difícil de predecir desde variables preop convencionales.

Un modelo que intenta predecir "cualquier complicación" debe aprender simultáneamente cuatro señales diferentes. Eso explica por qué el AUC global se queda en 0.75.

### 5.2 La predictibilidad varía por fenotipo

Cruzando los resultados del Enfoque C con los clusters:

| Cluster | Flag dominante | AUC individual del flag | Predecible? |
|---------|----------------|------------------------|-------------|
| C1 | `flag_uci_no_planeada`, `flag_hospitalizacion_no_anticipada` | 0.776 / 0.814 | **SÍ** |
| C2 | `flag_intubado_salida` | **0.552** | **NO** |
| C3 | `flag_hospitalizacion_no_anticipada` | 0.814 | **SÍ** |
| C4 | `flag_aferesis` | 0.595 | **Parcialmente** |

Los clusters C1 y C3 contienen eventos predecibles desde preop. El cluster C2 contiene principalmente un evento impredecible. El cluster C4 es parcialmente predecible. Un target construido excluyendo C2 y enfocándose en C1+C3 sería sustancialmente más predecible.

### 5.3 El Cluster C1 revela que la comorbilidad importa más que la complejidad del procedimiento para las complicaciones graves

Los pacientes de C1 (los más graves) tienen en realidad **menor score de complejidad del procedimiento** que C3 (hospitalizaciones), pero tienen el doble de HTA y casi el triple de diabetes. Esto desafía la intuición de que "los procedimientos más complejos causan más complicaciones" — para las complicaciones más graves (UCI), son las comorbilidades del paciente las que dominan.

Esto tiene implicaciones directas para la selección de features: las variables preop relacionadas con comorbilidades crónicas (HTA, diabetes, función renal, etc.) deberían tener mayor peso relativo en un modelo orientado a predecir complicaciones graves.

### 5.4 Propuesta de estrategia de modelado basada en clustering

En lugar de un solo modelo para un target compuesto, podría plantearse:

1. **Modelo A:** predice ingreso a UCI no planificado o hospitalización no anticipada (C1 + C3). Alta predictibilidad desde preop (AUC esperado > 0.80).
2. **Modelo B:** predice necesidad de aféresis o manejo hematológico intensivo (C4). Requeriría features específicas de historial hematológico.
3. **Descontinuar:** predecir `flag_intubado_salida` como parte del target (C2) — evento intraoperatorio no predecible.

Este enfoque de múltiples modelos especializados podría superar en performance al modelo único actual.
