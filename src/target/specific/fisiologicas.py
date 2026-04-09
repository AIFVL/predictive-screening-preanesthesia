import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_fisiologicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en variables fisiológicas prequirúrgicas con valores anormales.
    Realiza imputación de valores inválidos (nulos, ceros, valores fuera de límites fisiológicos)
    con valores normales antes de evaluar.
    
    Columnas agregadas:
        - flag_presion_sistolica_anormal
        - flag_presion_diastolica_anormal
        - flag_presion_media_anormal
        - flag_frecuencia_cardiaca_anormal
        - flag_saturacion_oxigeno_anormal
        - flag_temperatura_anormal
        - flag_glucometria_anormal
        - flag_fisiologicas (agregado)
    """
    df_result = df.copy()
    
    # Criterios y límites para variables fisiológicas
    criterios_fisiologicas = {
        'Presión arterial sistólica prequirúrgica ': {
            'flag': 'flag_presion_sistolica_anormal',
            'min': 80,
            'max': 150,
            'imputacion': 115,  # Valor normal medio
            'limite_invalido_min': 50,  # Físicamente imposible < 50 mmHg
            'limite_invalido_max': 200  # Físicamente imposible > 200 mmHg
        },
        'Presión arterial diastólica prequirúrgica': {
            'flag': 'flag_presion_diastolica_anormal',
            'min': 50,
            'max': 100,
            'imputacion': 75,  # Valor normal medio
            'limite_invalido_min': 20,  # Físicamente imposible < 20 mmHg
            'limite_invalido_max': 150  # Físicamente imposible > 150 mmHg
        },
        'Presión arterial media prequirúrgica': {
            'flag': 'flag_presion_media_anormal',
            'min': 60,
            'max': 115,
            'imputacion': 87.5,  # Valor normal medio
            'limite_invalido_min': 25,  # Físicamente imposible < 25 mmHg
            'limite_invalido_max': 200  # Físicamente imposible > 200 mmHg
        },
        'Frecuencia cardiaca prequirúrgica': {
            'flag': 'flag_frecuencia_cardiaca_anormal',
            'min': 50,
            'max': 110,
            'imputacion': 80,  # Valor normal medio
            'limite_invalido_min': 25,  # Físicamente imposible < 25 bpm
            'limite_invalido_max': 200  # Físicamente imposible > 200 bpm
        },
        'Saturación de oxígeno prequirúrgica': {
            'flag': 'flag_saturacion_oxigeno_anormal',
            'min': 95,
            'max': None,  # Solo mínimo
            'imputacion': 98,  # Valor normal
            'limite_invalido_min': 50,  # Físicamente imposible < 50%
            'limite_invalido_max': 100  # Máximo posible es 100%
        },
        'Temperatura prequirúrgica': {
            'flag': 'flag_temperatura_anormal',
            'min': 35,
            'max': 37.5,
            'imputacion': 36.75,  # Valor normal medio
            'limite_invalido_min': 30,  # Físicamente imposible < 30°C (hipotermia extrema incompatible con vida)
            'limite_invalido_max': 45  # Físicamente imposible > 45°C (hipertermia extrema incompatible con vida)
        }
    }
    
    # Aplicar imputación y criterios para variables con rango min-max
    for col, criterio in criterios_fisiologicas.items():
        serie = df_result[col].copy()
        
        # Identificar valores inválidos: nulos, ceros, o valores fuera de límites fisiológicos
        invalidos = (
            serie.isna() | 
            (serie == 0) | 
            (serie < criterio['limite_invalido_min']) | 
            (serie > criterio['limite_invalido_max'])
        )
        
        # Imputar valores inválidos con valor normal
        serie_imputada = serie.copy()
        serie_imputada[invalidos] = criterio['imputacion']
        
        # Replace previous column
        df_result = df_result.drop(columns=[col])
        df_result[col] = serie_imputada
        
        # Aplicar criterios de normalidad sobre la serie imputada
        if criterio['max'] is None:
            # Solo mínimo (saturación de oxígeno)
            df_result[criterio['flag']] = (serie_imputada < criterio['min']).astype(int)
        else:
            # Rango min-max
            df_result[criterio['flag']] = ((serie_imputada < criterio['min']) | (serie_imputada > criterio['max'])).astype(int)
    
    # Glucometría: valor = 1 (Sí) o valor fuera de rango normal (70-100 mg/dL)
    glucometria_flag = (df_result['Glucometría prequirúrgica '] == 1).astype(int)
    valor_glucometria = pd.to_numeric(df_result['Valor glucometria '], errors='coerce')
    
    # Imputar valores inválidos de glucometría (nulos, ceros, valores fuera de límites fisiológicos)
    # Límites: < 20 mg/dL (hipoglucemia extrema incompatible con vida) o > 600 mg/dL (hiperglucemia extrema)
    invalidos_glucometria = (
        valor_glucometria.isna() | 
        (valor_glucometria == 0) | 
        (valor_glucometria < 20) | 
        (valor_glucometria > 600)
    )
    valor_glucometria_imputado = valor_glucometria.copy()
    valor_glucometria_imputado[invalidos_glucometria] = 100  # Valor normal medio

    # Replace previous column
    df_result = df_result.drop(columns=['Valor glucometria '])
    df_result['Valor glucometria '] = valor_glucometria_imputado

    glucometria_valor_anormal = ((valor_glucometria_imputado < 70) | (valor_glucometria_imputado > 180)).astype(int)
    df_result['flag_glucometria_anormal'] = ((glucometria_flag == 1) | (glucometria_valor_anormal == 1)).astype(int)
    
    fisiologicas_flags = [
        'flag_presion_sistolica_anormal', 'flag_presion_diastolica_anormal', 'flag_presion_media_anormal',
        'flag_frecuencia_cardiaca_anormal', 'flag_saturacion_oxigeno_anormal', 'flag_temperatura_anormal',
        'flag_glucometria_anormal'
    ]
    df_result['flag_fisiologicas'] = (df_result[fisiologicas_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN VARIABLES FISIOLÓGICAS (VALORES ANORMALES):")
    print("  (Valores inválidos imputados con valores normales antes de evaluar)")
    for flag in fisiologicas_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_fisiologicas: {df_result['flag_fisiologicas'].sum():,} ({df_result['flag_fisiologicas'].mean()*100:.2f}%)")
    
    return df_result