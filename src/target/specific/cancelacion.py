import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_cancelacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en cancelación del procedimiento.
    """
    df_result = df.copy()

    df_result['canceladas'] = (df_result['Cirugia'].isna() & df_result['Otros procedimientos'].isna()).astype(int)

    df_result['canceladas_por_medico'] = (df_result['Cancelación de procedimiento - episodio consulta preanes'].str.contains('C-CA por Médico') | df_result['Cancelación de procedimiento - episodio procedimiento'].str.contains('C-CA por Médico')).astype(int)

    df_result['flag_cancelacion'] = (df_result['canceladas'] == 1) & (df_result['canceladas_por_medico'] == 1)

    print("\nVALIDACIÓN CANCELACION:")
    print(f"  Canceladas por medico: {df_result['canceladas_por_medico'].sum():,} ({df_result['canceladas_por_medico'].mean()*100:.2f}%)\n")

    # Drop previous columns
    df_result = df_result.drop(columns=['Cancelación de procedimiento - episodio consulta preanes', 'Cancelación de procedimiento - episodio procedimiento'])

    return df_result