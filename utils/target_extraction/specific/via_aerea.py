import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_via_aerea(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en variables de vía aérea (valores que indican complejidad).
    
    Columnas agregadas:
        - flag_intubacion_dificil (Dificil, Imposible)
        - flag_laringoscopia_alta (valores numéricos altos, percentil 75+)
        - flag_tipo_intubacion_complejo (Nasotraqueal, Otro, Traqueostomia)
        - flag_hoja_laringoscopio_recta (recta)
        - flag_elemento_via_aerea_complejo (Fibrolaringoscopio, Otro)
        - flag_via_aerea (agregado)
    """
    df_result = df.copy()
    
    # Intubación: Ninguna, Facil, Dificil, Imposible
    intubacion = df_result['Intubación  '].astype(str).str.strip().str.title()
    valores_intubacion_dificil = ['Dificil', 'Imposible']
    df_result['flag_intubacion_dificil'] = intubacion.isin(valores_intubacion_dificil).astype(int)
    
    # Laringoscopia: valores numéricos (valores altos indican mayor complejidad)
    laringoscopia = pd.to_numeric(df_result['Laringoscopia '], errors='coerce')
    df_result['flag_laringoscopia_alta'] = (laringoscopia > 1).astype(int)
    
    # Tipo intubación: Orotraqueal, Nasotraqueal, Ninguna, Otro, Traqueostomia
    tipo_intubacion = df_result['Tipo de intubación'].astype(str).str.strip().str.title()
    valores_tipo_complejo = ['Nasotraqueal', 'Otro', 'Traqueostomia']
    df_result['flag_tipo_intubacion_complejo'] = tipo_intubacion.isin(valores_tipo_complejo).astype(int)
    
    # Hoja de laringoscopio: Recta, curva, Ninguna
    hoja_laringoscopio = df_result['Hoja de laringoscopio '].astype(str).str.strip().str.lower()
    df_result['flag_hoja_laringoscopio_recta'] = (hoja_laringoscopio == 'recta').astype(int)
    
    # Elemento vía aérea: Tubo Endotraqueal, Canulanasal, Mascara Laringea, Fibrolaringoscopio, Mascara facial, Ninguno, Otro
    elemento_via_aerea = df_result['Elemento de vía aérea utilizado '].astype(str).str.strip().str.title()
    valores_elemento_complejo = ['Fibrolaringoscopio', 'Otro']
    df_result['flag_elemento_via_aerea_complejo'] = elemento_via_aerea.isin(valores_elemento_complejo).astype(int)
    
    # Agregar flag si hay al menos un indicador de complejidad
    via_aerea_flags = [
        'flag_intubacion_dificil', 'flag_laringoscopia_alta', 'flag_tipo_intubacion_complejo',
        'flag_hoja_laringoscopio_recta', 'flag_elemento_via_aerea_complejo'
    ]
    df_result['flag_via_aerea'] = (df_result[via_aerea_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN VÍA AÉREA (COMPLEJIDAD):")
    for flag in via_aerea_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_via_aerea (≥1 indicador): {df_result['flag_via_aerea'].sum():,} ({df_result['flag_via_aerea'].mean()*100:.2f}%)")
    
    return df_result