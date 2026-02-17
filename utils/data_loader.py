import pandas as pd
import os

def load_opera_completo():
    """
    Carga el dataset 'OPERA_COMPLETO.xlsx' intentando varias rutas posibles.
    Retorna datos (df) o lanza FileNotFoundError.
    """
    print(f"Directorio actual: {os.getcwd()}")
    print("Cargando dataset completo (OPERA_COMPLETO.xlsx)...")
    
    paths_to_try = [
        '../OPERA_COMPLETO.xlsx',
        'OPERA_COMPLETO.xlsx',
        'c:/Users/Usuario/Desktop/predictive-screening-preanesthesia/OPERA_COMPLETO.xlsx'
    ]
    
    for path in paths_to_try:
        try:
            df = pd.read_excel(path)
            print(f"✅ Dataset cargado desde: {path}")
            print(f"   Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            
            if 'target' in df.columns:
                print(f"   Proporción target=1: {df['target'].mean()*100:.2f}%")
            else:
                raise ValueError("¡ERROR CRÍTICO! La columna 'target' no existe en el Excel cargado.")
                
            return df
        except FileNotFoundError:
            continue
            
    raise FileNotFoundError(f"No se encontró OPERA_COMPLETO.xlsx en ninguna de las rutas: {paths_to_try}")

def load_features_metadata():
    """
    Carga 'variables_seleccionadas.csv' intentando varias rutas.
    """
    paths_to_try = [
        '../features/variables_seleccionadas.csv',
        'features/variables_seleccionadas.csv',
        'c:/Users/Usuario/Desktop/predictive-screening-preanesthesia/features/variables_seleccionadas.csv'
    ]
    
    for path in paths_to_try:
        try:
            df_meta = pd.read_csv(path)
            print(f"✅ Metadatos de features cargados desde: {path}")
            return df_meta
        except FileNotFoundError:
            continue
            
    raise FileNotFoundError(f"No se encontró variables_seleccionadas.csv en ninguna de las rutas: {paths_to_try}")
