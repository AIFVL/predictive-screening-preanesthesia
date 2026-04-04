import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_reservas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en reservas de hemoderivados y sangre.
    
    Columnas agregadas:
        - flag_reserva_hemoderivados (código 912002)
        - flag_reserva_sangre (valor = 1)
        - flag_reservas (agregado)
    """
    df_result = df.copy()
    
    df_result['flag_reserva_hemoderivados'] = (df_result['Reserva de hemoderivados'] == 912002).astype(int)
    df_result['flag_reserva_sangre'] = (df_result['Reserva de sangre '] == 1).astype(int)
    df_result['flag_reservas'] = ((df_result['flag_reserva_hemoderivados'] == 1) | (df_result['flag_reserva_sangre'] == 1)).astype(int)
    
    print("VALIDACIÓN RESERVAS:")
    print_validation_result('flag_reserva_hemoderivados', df_result['flag_reserva_hemoderivados'], len(df_result))
    print_validation_result('flag_reserva_sangre', df_result['flag_reserva_sangre'], len(df_result))
    print(f"\n  TOTAL flag_reservas: {df_result['flag_reservas'].sum():,} ({df_result['flag_reservas'].mean()*100:.2f}%)")

    # Drop previous columns
    df_result = df_result.drop(columns=['Reserva de hemoderivados', 'Reserva de sangre '])
    
    return df_result