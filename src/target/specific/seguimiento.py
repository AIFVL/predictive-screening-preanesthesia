import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_seguimiento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en seguimiento post-operatorio.
    
    Variables evaluadas:
        - Consulta a urgencias en los siguientes 30 días
        - Necesidad de interconsultas a otras especialidades médicas

    Columnas agregadas:
        - flag_urgencias_30_dias (tiene fecha)
        - flag_interconsultas (marcada con 'x')
        - flag_seguimiento (agregado)
    """
    df_result = df.copy()
    
    # Consulta urgencias 30 días: tiene fecha
    df_result['flag_urgencias_30_dias'] = df_result['Consulta a urgencias en los siguientes 30 días'].notna().astype(int)
    
    # Interconsultas necesarias: marcada con 'x'
    df_result['flag_interconsultas'] = (
        df_result['Necesidad de interconsultas a otras especialidades médicas'].str.lower().str.strip() == 'x'
    ).astype(int)
    
    # flag_interconsultas excluido del target: las interconsultas pueden ser parte del manejo
    # rutinario y planeado (ej. oncología solicitando manejo del dolor postoperatorio)
    # Solo la consulta a urgencias en 30 días es un indicador validado de reingreso no planeado
    # flag_interconsultas sigue calculado en el dataframe como posible feature predictivo
    seguimiento_flags = ['flag_urgencias_30_dias']
    df_result['flag_seguimiento'] = (df_result[seguimiento_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN SEGUIMIENTO POST-OPERATORIO:")
    for flag in seguimiento_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_seguimiento: {df_result['flag_seguimiento'].sum():,} ({df_result['flag_seguimiento'].mean()*100:.2f}%)")
    
    return df_result

