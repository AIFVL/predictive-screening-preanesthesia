import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_desenlace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en indicadores de desenlace del procedimiento.
    
    Variables evaluadas:
        - Destino final
        - Estado intubación
        - Fallecimiento durante cirugía (estado de consciencia)
        - Complicaciones
    
    Columnas agregadas:
        - flag_destino_uci (destino UCI u Otro)
        - flag_intubado_salida (salió intubado)
        - flag_no_despierto (no despertó adecuadamente)
        - flag_complicacion (presentó complicación)
        - flag_desenlace (agregado)
    """
    df_result = df.copy()
    
    # Flag: Destino UCI u Otro (no UCPA ni AMB)
    destino = df_result['Destino final '].fillna('').astype(str).str.strip().str.upper()
    df_result['flag_destino_uci'] = destino.isin(['UCI', 'OTRO']).astype(int)
    
    # Flag: Salió intubado
    estado_int = df_result['Estado intubación '].fillna('').astype(str).str.strip().str.title()
    df_result['flag_intubado_salida'] = (estado_int == 'Intubado').astype(int)
    
    # Flag: No despertó adecuadamente (Dormido, no Despierto ni Reactivo)
    estado_consc = df_result['Fallecimiento durante cirugía '].fillna('').astype(str).str.strip()
    # Limpiar caracteres especiales como \t
    estado_consc = estado_consc.str.replace(r'^\s+', '', regex=True)

    # Replace previous column
    df_result = df_result.drop(columns=['Fallecimiento durante cirugía '])
    df_result['Fallecimiento durante cirugía '] = estado_consc

    despierto = estado_consc.str.lower().isin(['despierto', 'reactivo'])
    tiene_dato = estado_consc != ''
    df_result['flag_no_despierto'] = (tiene_dato & ~despierto).astype(int)
    
    # Flag: Complicación registrada
    complicaciones = df_result['Complicaciones '].fillna('').astype(str).str.strip()
    df_result['flag_complicacion'] = (complicaciones == 'X').astype(int)
    
    # Flag agregado
    desenlace_flags = [
        'flag_destino_uci', 'flag_intubado_salida', 'flag_no_despierto', 'flag_complicacion'
    ]
    df_result['flag_desenlace'] = (df_result[desenlace_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN DESENLACE:")
    for flag in desenlace_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_desenlace: {df_result['flag_desenlace'].sum():,} ({df_result['flag_desenlace'].mean()*100:.2f}%)")
    
    return df_result
