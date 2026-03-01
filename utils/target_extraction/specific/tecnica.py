import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_tecnica(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en técnica anestésica y monitoreo.
    
    Variables evaluadas:
        - Técnica anestésica
        - Monitoreo
    
    Columnas agregadas:
        - flag_anestesia_general (contiene 'General')
        - flag_tecnica_combinada (técnica combinada con ',')
        - flag_monitoreo_invasivo (monitoreo avanzado)
        - flag_tecnica (agregado)
    """
    df_result = df.copy()
    
    # Normalizar técnica anestésica
    tecnica = df_result['Técnica anestésica '].fillna('').astype(str)
    
    # Flag: Técnica combinada (más de una técnica listada, separado por coma)
    df_result['flag_tecnica_combinada'] = tecnica.str.count(',').gt(1).astype(int)
    
    # Normalizar monitoreo
    monitoreo = df_result['Monitoreo '].fillna('').astype(str)
    
    # Flag: Monitoreo invasivo/avanzado (Temperatura o Analizador de gases)
    keywords = ['Gases Sanguineos', 'PVC','Temperatura']
    df_result['flag_monitoreo_invasivo'] = 0
    
    for keyword in keywords:
        tiene_keyword = monitoreo.str.contains(keyword, case=False, na=False)
        df_result['flag_monitoreo_invasivo'] = tiene_keyword.astype(int) | df_result['flag_monitoreo_invasivo']
    
    # Flag agregado
    tecnica_flags = ['flag_tecnica_combinada', 'flag_monitoreo_invasivo']
    df_result['flag_tecnica'] = (df_result[tecnica_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN TÉCNICA ANESTÉSICA:")
    for flag in tecnica_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_tecnica: {df_result['flag_tecnica'].sum():,} ({df_result['flag_tecnica'].mean()*100:.2f}%)")
    
    return df_result
