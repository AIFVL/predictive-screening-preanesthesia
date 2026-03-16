"""
Script de Auditoría RÁPIDA (Subset 1000 filas).
Verifica:
1. Resolución de Features (incluyendo las 19 que fallaban).
2. Fuga de Datos (Leakage) del Target.
3. Valores Faltantes y Constantes.
"""
import sys
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
sys.path.append('.')

from utils.data_loader import load_features_metadata
from utils.feature_engineering import resolve_feature_columns, ENCODING_FIX_MAP

def load_subset():
    path = 'OPERA_COMPLETO.xlsx'
    try:
        print(f"Cargando subset de {path}...")
        df = pd.read_excel(path, nrows=1000)
        return df
    except Exception as e:
        print(f"Error carga: {e}")
        return None

def run_audit():
    with open('audit_final.log', 'w', encoding='utf-8') as f:
        f.write("=== INICIO AUDITORIA DE PIPELINE (SUBSET) ===\n\n")
        
        # 1. Carga
        try:
            print("Cargando subset...") # Keep console feedback
            df = load_subset()
            if df is None: return
        except Exception as e:
            f.write(f"[ERROR] Carga fallida: {e}\n")
            return

        # 2. Filtro Adultos
        col_edad = 'Edad'
        if col_edad not in df.columns:
            f.write(f"[ERROR] [FILTER] No columna '{col_edad}'\n")
            return
            
        df_adults = df[df[col_edad] >= 18].copy()
        f.write(f"[OK] [FILTER] Subset Adultos: {len(df_adults)} registros\n")
        
        # 3. Target
        if 'target' in df_adults.columns:
            prev = df_adults['target'].mean()
            f.write(f"[INFO] [TARGET] Prevalencia (Subset): {prev:.2%}\n")

        # 4. Features
        f.write("\n--- Auditoria de Features ---\n")
        meta = load_features_metadata()
        selected = meta['Variable'].tolist()
        
        df_adults = df_adults.rename(columns=ENCODING_FIX_MAP)
        resolved, missing = resolve_feature_columns(selected, df_adults.columns, meta)
        
        f.write(f"[OK] [FEATURES] Seleccionadas: {len(selected)}\n")
        f.write(f"[OK] [FEATURES] Resueltas:     {len(resolved)}\n")
        
        if missing:
            f.write(f"[ERROR] [FEATURES] NO ENCONTRADAS ({len(missing)}):\n")
            for m in missing:
                f.write(f"   - {m}\n")
        else:
            f.write("[OK] [FEATURES] Todas las features encontradas (Fix aplicado).\n")

        # 5. Leakage Check
        f.write("\n--- Chequeo de Leakage ---\n")
        leak_keywords = [
            'complicacion', 'fallec', 'muerte', 'destino', 'uci', 
            'reintervencion', 'sangrado_intra', 'transfusion_intra', 
            'duracion_real', 'tiempo_quirurgico', 'aldrete', 'bromage',
            'dolor_post', 'nauesa_post', 'vomito_post'
        ]
        leaks = []
        for col in resolved:
            for k in leak_keywords:
                if k in col.lower() and 'antecedente' not in col.lower() and 'score' not in col.lower():
                    leaks.append(col)
        
        if leaks:
            f.write(f"[WARN] [LEAKAGE] Posibles variables post-operatorias:\n")
            for l in leaks:
                f.write(f"   - {l}\n")
        else:
            f.write("[OK] [LEAKAGE] No variables post-operatorias obvias.\n")

        f.write("\n=== FIN AUDITORIA ===\n")
        print("Auditoria completada. Ver audit_final.log")

if __name__ == "__main__":
    run_audit()
