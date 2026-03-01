import pandas as pd
import os

def load_opera_completo(path: str):
    """
    Carga el dataset 'OPERA_COMPLETO.xlsx' intentando varias rutas posibles.
    Retorna datos (df) o lanza FileNotFoundError.
    """
    print(f"Directorio actual: {os.getcwd()}")
    print("Cargando dataset completo (OPERA_COMPLETO.xlsx)...")
    
    df = pd.read_excel(path)
    print(f"✅ Dataset cargado desde: {path}")
    print(f"   Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    if 'target' in df.columns:
        print(f"   Proporción target=1: {df['target'].mean()*100:.2f}%")
    else:
        raise ValueError("¡ERROR CRÍTICO! La columna 'target' no existe en el Excel cargado.")
        
    return df

def load_features_metadata(path: str):
    """
    Carga 'variables_seleccionadas.csv' intentando varias rutas.
    """
    df_meta = pd.read_csv(path)
    print(f"✅ Metadatos de features cargados desde: {path}")
    print(f"   Dimensiones: {df_meta.shape[0]:,} filas × {df_meta.shape[1]} columnas")
    return df_meta
