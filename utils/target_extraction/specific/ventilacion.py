import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_ventilacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en parámetros de ventilación mecánica.
    
    Variables evaluadas:
        - Ventilación espontánea, Ventilación asistida, Control manual
        - Control volumen, Presión soporte, Control ventilatorio, Control presión
        - SIMV/PS, VT, PEEP, Frecuencia respiratoria
    
    Columnas agregadas:
        - flag_no_ventilacion_espontanea (no tiene ventilación espontánea)
        - flag_ventilacion_asistida (requirió asistencia)
        - flag_control_manual (control manual)
        - flag_modos_avanzados (SIMV/PS o Presión soporte)
        - flag_parametros_ventilatorios (≥3 parámetros marcados)
        - flag_frecuencia_resp_anormal (FR <8 o >25 rpm)
        - flag_ventilacion (agregado)
    """
    df_result = df.copy()
    
    # Columnas de modos ventilatorios (todas tienen 'X' si están marcadas)
    modos_cols = [
        'Control volumen', 'Presión soporte ', 'Control ventilatorio ',
        'Control presión ', 'SIMV/PS', 'VT', 'PEEP'
    ]
    
    # Flag: Ventilación asistida
    df_result['flag_ventilacion_asistida'] = (
        df_result['Ventilación asistida '].fillna('').str.strip() == 'X'
    ).astype(int)
    
    # Flag: Control manual
    df_result['flag_control_manual'] = (
        df_result['Control manual '].fillna('').str.strip() == 'X'
    ).astype(int)
    
    # Flag: Modos avanzados (SIMV/PS o Presión soporte)
    simv = df_result['SIMV/PS'].fillna('').str.strip() == 'X'
    presion_soporte = df_result['Presión soporte '].fillna('').str.strip() == 'X'
    df_result['flag_modos_avanzados'] = (simv | presion_soporte).astype(int)
    
    # Flag: Múltiples parámetros ventilatorios (≥4 marcados)
    contador_modos = pd.DataFrame()
    for col in modos_cols:
        if col in df_result.columns:
            contador_modos[col] = (df_result[col].fillna('').str.strip() == 'X').astype(int)
    df_result['flag_parametros_ventilatorios'] = (contador_modos.sum(axis=1) >= 4).astype(int)
    
    # Flag: Frecuencia respiratoria anormal (<8 o >25, excluyendo 0)
    fr = pd.to_numeric(df_result['Frecuencia respiratoria '], errors='coerce')
    fr_valida = fr > 0  # Excluir 0 que parece ser valor por defecto
    fr_anormal = (fr < 8) | (fr > 25)
    df_result['flag_frecuencia_resp_anormal'] = (fr_valida & fr_anormal).astype(int)
    
    # Flag agregado
    ventilacion_flags = [
        'flag_ventilacion_asistida', 'flag_control_manual',
        'flag_modos_avanzados', 'flag_parametros_ventilatorios', 'flag_frecuencia_resp_anormal'
    ]
    df_result['flag_ventilacion'] = (df_result[ventilacion_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN VENTILACIÓN:")
    for flag in ventilacion_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_ventilacion: {df_result['flag_ventilacion'].sum():,} ({df_result['flag_ventilacion'].mean()*100:.2f}%)")
    
    return df_result
