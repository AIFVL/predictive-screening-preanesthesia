import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_estancia(df: pd.DataFrame, umbral_estancia: int = 3) -> pd.DataFrame:
    """
    Valida basado en estancia hospitalaria y UCI.
    
    Parámetros:
        - umbral_estancia: días para considerar estancia prolongada (default: 3)

    Variables evaluadas:
        - Estancia hospitalaria
        - Hospitalización no anticipada
        - Admisión a cuidado intensivo no planeada
        - Estancia en UCI
    
    Columnas agregadas:
        - flag_estancia_prolongada (> umbral días)
        - flag_hospitalizacion_no_anticipada (marcada con 'x')
        - flag_uci_no_planeada (tiene fecha)
        - flag_estancia_uci (tiene valor)
        - flag_estancia (agregado)
    """
    df_result = df.copy()
    
    # Estancia hospitalaria: convertir a numérico (amb = 0, valores numéricos = días)
    estancia = df_result['Estancia hospitalaria '].copy()
    estancia = estancia.replace({'amb': 0})
    estancia = pd.to_numeric(estancia, errors='coerce').fillna(0)

    # Replace previous column
    df_result = df_result.drop(columns=['Estancia hospitalaria '])
    df_result['Estancia hospitalaria '] = estancia

    df_result['flag_estancia_prolongada'] = (estancia > umbral_estancia).astype(int)
    
    # Hospitalización no anticipada: marcada con 'x'
    df_result['flag_hospitalizacion_no_anticipada'] = (
        df_result['Hospitalización no anticipada'].str.lower().str.strip() == 'x'
    ).astype(int)
    
    # Admisión UCI no planeada: tiene fecha
    df_result['flag_uci_no_planeada'] = df_result['Admisión a cuidado intensivo no planeada'].notna().astype(int)
    
    # Estancia en UCI: tiene valor
    df_result['flag_estancia_uci'] = df_result['Estancia en UCI '].notna().astype(int)
    
    estancia_flags = [
        'flag_estancia_prolongada', 'flag_hospitalizacion_no_anticipada',
        'flag_uci_no_planeada', 'flag_estancia_uci'
    ]
    df_result['flag_estancia'] = (df_result[estancia_flags].sum(axis=1) > 0).astype(int)
    
    print(f"\nVALIDACIÓN ESTANCIA (umbral: >{umbral_estancia} días):")
    for flag in estancia_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_estancia: {df_result['flag_estancia'].sum():,} ({df_result['flag_estancia'].mean()*100:.2f}%)")
    
    return df_result

