from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawField:
    name: str         # columna exacta del dataset (incluye tildes, espacios, paréntesis)
    dtype: str        # "object", "float64"
    description: str  # descripción legible para el frontend


# Fuente única de verdad: los campos clínicos crudos que acepta la API.
# El orden define el orden en que el frontend renderiza el formulario.
RAW_INPUT_SCHEMA: list[RawField] = [
    # ── Datos demográficos ──────────────────────────────────────────────────
    RawField("Edad", "float64", "Edad del paciente (años)"),
    RawField("Sexo", "object", 'Sexo biológico — valores: "M" o "F"'),
    RawField("Atención", "object", 'Tipo de atención — valores: "Electivo", "Urgencia", "Urgencia Programada"'),
    # ── Antropometría y signos vitales ──────────────────────────────────────
    RawField("Peso (Kg)", "float64", "Peso corporal (kg)"),
    RawField("Talla (cm)", "float64", "Talla (cm)"),
    RawField("IMC", "float64", "Índice de masa corporal (kg/m²)"),
    RawField("Tensión Arterial Sistólica (mm/Hg)", "float64", "Presión arterial sistólica (mmHg)"),
    RawField("Tensión Arterial Diastólica (mm/Hg)", "float64", "Presión arterial diastólica (mmHg)"),
    RawField("Tensión Arterial Media (mm/Hg)", "float64", "Presión arterial media (mmHg)"),
    RawField("Frecuencia Respiratoria", "float64", "Frecuencia respiratoria (resp/min)"),
    RawField("Temperatura", "float64", "Temperatura corporal (°C)"),
    # ── Fecha / hora ────────────────────────────────────────────────────────
    RawField("Fecha", "object", "Fecha de valoración preanestésica (YYYY-MM-DD)"),
    RawField("Hora", "object", "Hora de valoración preanestésica (HH:MM)"),
    # ── Laboratorio ─────────────────────────────────────────────────────────
    RawField("Grupo Sanguíneo", "object", 'Grupo sanguíneo ABO — valores: "A", "B", "AB", "O"'),
    RawField("RH", "object", 'Factor RH — valores: "Positivo", "Negativo"'),
    RawField("Examen_Hemoglobina(g/dl)", "float64", "Hemoglobina (g/dL)"),
    RawField("Examen_PT (INR)", "float64", "Tiempo de protrombina — INR"),
    # ── Antecedentes ────────────────────────────────────────────────────────
    RawField("Antecedente neurológicos", "object", "Antecedentes neurológicos (texto libre; p. ej. epilepsia, ACV)"),
    RawField("Antecedente respiratorios", "object", "Antecedentes respiratorios (texto libre; p. ej. asma, EPOC)"),
    RawField("Antecedentes cardiovasculares", "object", "Antecedentes cardiovasculares (texto libre; p. ej. HTA, FA)"),
    RawField("Antecedente hematológicos ", "object", "Antecedentes hematológicos (texto libre)"),
    RawField("Antecedente endocrinológicos", "object", "Antecedentes endocrinológicos (texto libre; p. ej. DM2, hipotiroidismo)"),
    RawField("Antecedente renales", "object", "Antecedentes renales (texto libre)"),
    RawField("Antecedente gastrointestinales", "object", "Antecedentes gastrointestinales (texto libre)"),
    RawField("Antecedentes anestésicos", "object", "Antecedentes anestésicos (texto libre; p. ej. intubación difícil)"),
    RawField("Antecedentes quirúrgicos", "object", "Antecedentes quirúrgicos (texto libre; p. ej. apendicectomía 2015)"),
    RawField("Anestesia previa", "object", "Tipo de anestesia en cirugías previas (texto libre; p. ej. general, raquidea)"),
    # ── Examen físico — via aérea ───────────────────────────────────────────
    RawField("Clase Funcional (NYHA) ", "object", 'Clase funcional NYHA — valores: "I", "II", "III", "IV"'),
    RawField("Cuello Móvil", "object", 'Movilidad cervical — valores: "Normal", "Reducido", "Fijo"'),
    RawField("Apertura Oral", "object", 'Apertura oral — valores: "> 3 cm"  o  "< 3 cm"'),
    RawField("Puntaje Mallampati", "float64", "Clasificación Mallampati — valor numérico: 1, 2, 3 o 4"),
    RawField("Estado Nutricional", "object", 'Estado nutricional — valores: "Normal", "Bajo peso", "Sobrepeso", "Obesidad" (se recalcula desde IMC si está vacío)'),
    RawField("Color de Piel", "object", "Color de piel (texto libre)"),
    # ── Escala de Glasgow ────────────────────────────────────────────────────
    RawField("Sistema Nervioso", "object", 'Sistema nervioso — p. ej. "normal", "somnolencia", "confuso"'),
    RawField("Apertura Visual", "object", 'Glasgow — apertura ocular: "espontanea", "a la voz", "al dolor", "ninguna"'),
    RawField("Respuesta Verbal", "object", 'Glasgow — respuesta verbal: "conversacion orientada", "conversacion confusa", "palabras inapropiadas", "ruidos incomprensibles", "ninguna"'),
    RawField("Respuesta Motora", "object", 'Glasgow — respuesta motora: "obedece ordenes", "localiza el dolor", "retira al dolor", "flexiona al dolor", "flacido"'),
    # ── Examen físico — sistemas ─────────────────────────────────────────────
    RawField("Sistema Respiratorio", "object", 'Hallazgos respiratorios (texto libre; p. ej. "normal", "disnea, roncus")'),
    RawField("Sistema cardiovascular", "object", 'Hallazgos cardiovasculares (texto libre; p. ej. "normal", "soplo sistólico")'),
    RawField("Abdomen", "object", 'Hallazgos abdominales (texto libre; p. ej. "normal", "distendido")'),
    RawField("Arritmia", "object", 'Arritmia documentada — valores: "Si" o "No"'),
    RawField("Angina", "object", 'Angina documentada — escriba "x" si hay angina, deje vacío si no'),
    RawField("Condición", "object", 'Condición general del paciente (texto libre; p. ej. "lúcido, orientado")'),
    # ── Plan quirúrgico ─────────────────────────────────────────────────────
    RawField("Tipo de anestesia propuesta", "object", 'Tipo de anestesia (texto libre; p. ej. "general", "raquidea", "general, epidural")'),
    RawField("Prótesis Dental", "object", 'Prótesis dental — valores: "Si" o "No"'),
    RawField("Alérgeno", "object", "Alérgenos conocidos (texto libre; p. ej. penicilina, látex)"),
    RawField("Dx Preoperatorio", "object", "Diagnóstico preoperatorio (texto libre)"),
    RawField("Procedimiento propuesto", "object", "Procedimiento quirúrgico propuesto (texto libre)"),
]


def raw_input_schema_as_dict() -> list[dict]:
    """Serializable para embed en el manifest JSON."""
    return [
        {"name": f.name, "dtype": f.dtype, "description": f.description}
        for f in RAW_INPUT_SCHEMA
    ]
