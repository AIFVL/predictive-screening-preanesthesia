EXCLUDED_FLAGS = [
    "flag_fisiologicas",
    "flag_ventilacion",
    # Excluidos por ser intervenciones planeadas o precauciones, no complicaciones:
    "flag_tecnica",    # Técnica combinada/monitoreo avanzado: decisión clínica planeada
    "flag_tiempos",    # Duración larga: puede ser cirugía compleja prevista
    "flag_induccion",  # Inducción mixta: elección técnica del anestesiólogo
    "flag_reservas",   # Reserva de sangre: precaución; transfusión real capturada en flag_liquidos
]

RELEVANT_FLAGS = [
    "flag_cancelacion",
    "flag_via_aerea",
    "flag_liquidos",
    "flag_desenlace",
    "flag_complicaciones_medicas",
    "flag_estancia",
    "flag_seguimiento",
]

ALL_FLAGS = [
    "flag_cancelacion",
    "flag_reservas",
    "flag_fisiologicas",
    "flag_tiempos",
    "flag_induccion",
    "flag_via_aerea",
    "flag_ventilacion",
    "flag_tecnica",
    "flag_liquidos",
    "flag_desenlace",
    "flag_complicaciones_medicas",
    "flag_estancia",
    "flag_seguimiento",
]
