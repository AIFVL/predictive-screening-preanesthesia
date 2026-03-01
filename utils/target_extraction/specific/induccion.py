import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_induccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en clase de inducción (valores que indican mayor complejidad).
    
    Valores posibles: Intravenosa, Ninguna, Inhalatoria, Mixta, o vacío
    
    Columnas agregadas:
        - flag_induccion_compleja (Mixta, Inhalatoria)
        - flag_induccion (agregado)
    """
    df_result = df.copy()
    
    # Normalizar valores (case insensitive, strip)
    clase_induccion = df_result['Clase de inducción '].astype(str).str.strip().str.title()
    
    # Valores que indican mayor complejidad
    valores_complejos = ['Mixta']
    
    df_result['flag_induccion_compleja'] = clase_induccion.isin(valores_complejos).astype(int)
    df_result['flag_induccion'] = df_result['flag_induccion_compleja']
    
    print("\nVALIDACIÓN CLASE DE INDUCCIÓN:")
    print(f"  Valores complejos considerados: {valores_complejos}")
    print_validation_result('flag_induccion_compleja', df_result['flag_induccion_compleja'], len(df_result))
    print(f"\n  TOTAL flag_induccion: {df_result['flag_induccion'].sum():,} ({df_result['flag_induccion'].mean()*100:.2f}%)")
    
    return df_result