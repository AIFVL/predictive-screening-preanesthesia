import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_complicaciones_medicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en complicaciones médicas postoperatorias (códigos CIE-10).
    
    Variables evaluadas:
        - Infarto de miocardio en hospital
        - Accidente cerebrovascular isquémico
        - Accidente cerebrovascular hemorrágico
        - Ataque isquémico transitorio (TIA)
        - Aspiración pulmonar
        - Complicaciones pulmonares posoperatorias

    Columnas agregadas:
        - flag_infarto_miocardio (I214, I219)
        - flag_acv_isquemico (I630-I639, I640)
        - flag_acv_hemorragico (I618, I619)
        - flag_tia (G453, G454, G458, G459)
        - flag_aspiracion_pulmonar (J690)
        - flag_complicaciones_pulmonares  (129, J159, J189, J958, J960, J988, R060, R061, T884)
        - flag_complicaciones_medicas (agregado)
    """
    df_result = df.copy()

    # Flag = 1 si tiene código CIE-10
    df_result['flag_infarto_miocardio'] = df_result['Infarto de miocardio en hospital'].notna().astype(int)
    df_result['flag_acv_isquemico'] = df_result['Accidente cerebrovascular isquémico'].notna().astype(int)
    df_result['flag_acv_hemorragico'] = df_result['Accidente cerebrovascular hemorrágico'].notna().astype(int)
    df_result['flag_tia'] = df_result['Ataque isquémico transitorio (TIA)'].notna().astype(int)
    df_result['flag_aspiracion_pulmonar'] = df_result['Aspiración pulmonar'].notna().astype(int)
    df_result['flag_complicaciones_pulmonares'] = df_result['Complicaciones pulmonares posoperatorias '].notna().astype(int)

    complicaciones_flags = [
        'flag_infarto_miocardio', 'flag_acv_isquemico', 'flag_acv_hemorragico', 'flag_tia',
        'flag_aspiracion_pulmonar', 'flag_complicaciones_pulmonares'
    ]
    df_result['flag_complicaciones_medicas'] = (df_result[complicaciones_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN COMPLICACIONES MÉDICAS (CIE-10):")
    for flag in complicaciones_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_complicaciones_medicas: {df_result['flag_complicaciones_medicas'].sum():,} ({df_result['flag_complicaciones_medicas'].mean()*100:.2f}%)")
    
    return df_result