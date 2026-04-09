import pandas as pd

from ..helpers import calculate_duration, print_validation_result

def validate_based_on_tiempos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en tiempos de procedimiento con duraciones anormalmente largas.
    
    Columnas agregadas:
        - duracion_cirujano_minutos
        - duracion_anestesia_minutos
        - flag_duracion_cirujano_larga (percentil 90)
        - flag_duracion_anestesia_larga (percentil 90)
        - flag_tiempos (agregado)
    """
    df_result = df.copy()
    
    df_result['duracion_cirujano_minutos'] = calculate_duration(
        df_result['Inicio cirujano '], df_result['Fin cirujano ']
    )
    df_result['duracion_anestesia_minutos'] = calculate_duration(
        df_result['Inicio anestesia '], df_result['Fin anestesia ']
    )
    
    # Duraciones anormalmente largas (percentil 90)
    p90_cirujano = df_result['duracion_cirujano_minutos'].quantile(0.90)
    p90_anestesia = df_result['duracion_anestesia_minutos'].quantile(0.90)
    
    df_result['flag_duracion_cirujano_larga'] = (df_result['duracion_cirujano_minutos'] > p90_cirujano).astype(int)
    df_result['flag_duracion_anestesia_larga'] = (df_result['duracion_anestesia_minutos'] > p90_anestesia).astype(int)
    
    tiempos_flags = ['flag_duracion_cirujano_larga', 'flag_duracion_anestesia_larga']
    df_result['flag_tiempos'] = (df_result[tiempos_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN TIEMPOS (DURACIONES LARGAS):")
    print(f"  Percentil 90 cirujano: {p90_cirujano:.1f} minutos")
    print(f"  Percentil 90 anestesia: {p90_anestesia:.1f} minutos")
    for flag in tiempos_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_tiempos: {df_result['flag_tiempos'].sum():,} ({df_result['flag_tiempos'].mean()*100:.2f}%)")
    
    return df_result