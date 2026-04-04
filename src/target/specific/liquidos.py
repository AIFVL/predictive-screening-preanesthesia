import pandas as pd

from ..helpers import print_validation_result

def validate_based_on_liquidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida basado en manejo de líquidos.
    
    Variables evaluadas:
        - Líquidos administrados, VOLUMEN LIQUIDOS ADMINISTRADOS
        - Liquidos eliminados, Volumen de liquidos eliminados
        - Balance de líquidos
    
    Columnas agregadas:
        - flag_hemoderivados (se administraron hemoderivados)
        - flag_aferesis (aféresis o productos sanguíneos)
        - flag_volumen_alto (volumen administrado > 1000 mL)
        - flag_perdidas_altas (pérdidas > 500 mL)
        - flag_balance_extremo (balance < -500 o > 1500 mL)
        - flag_liquidos (agregado)
    """
    df_result = df.copy()
    
    # Flag: Hemoderivados administrados
    liq_admin = df_result['Líquidos administrados '].fillna('').astype(str).str.lower()
    df_result['flag_hemoderivados'] = liq_admin.str.contains('hemoderivados', na=False).astype(int)
    
    # Flag: Aféresis o productos sanguíneos eliminados
    liq_elim = df_result['Liquidos eliminados'].fillna('').astype(str).str.lower()
    productos_sangre = liq_elim.str.contains('aferesis|plaquetas|plasma|crioprecipitados', na=False, regex=True)
    df_result['flag_aferesis'] = productos_sangre.astype(int)
    
    # Flag: Volumen alto administrado (> 1000 mL)
    vol_admin = pd.to_numeric(df_result['VOLUMEN LIQUIDOS ADMINISTRADOS'], errors='coerce').fillna(0)
    df_result['flag_volumen_alto'] = (vol_admin > 1000).astype(int)

    df_result = df_result.drop(columns=['VOLUMEN LIQUIDOS ADMINISTRADOS'])
    df_result['VOLUMEN LIQUIDOS ADMINISTRADOS'] = vol_admin
    
    # Flag: Pérdidas altas (> 1000 mL — hemorragia significativa en cirugía mayor)
    # Umbral elevado de 500 mL: pérdidas ≤1000 mL son fisiológicamente normales en cx mayores
    vol_elim = pd.to_numeric(df_result['Volumen de liquidos eliminados '], errors='coerce').fillna(0)
    df_result['flag_perdidas_altas'] = (vol_elim > 1000).astype(int)

    df_result = df_result.drop(columns=['Volumen de liquidos eliminados '])
    df_result['Volumen de liquidos eliminados '] = vol_elim

    # Flag: Balance extremo (< -1000 o > 2000 mL)
    # Umbrales elevados: reposición agresiva en cx mayor es normal; se captura solo desequilibrio real
    balance = pd.to_numeric(df_result['Balance de líquidos '], errors='coerce').fillna(0)
    df_result['flag_balance_extremo'] = ((balance < -1000) | (balance > 2000)).astype(int)

    df_result = df_result.drop(columns=['Balance de líquidos '])
    df_result['Balance de líquidos '] = balance

    # Flag agregado
    # flag_volumen_alto excluido del target: >1000 mL administrados es estándar en cx mayor
    # (sigue calculado en el dataframe para uso como feature predictivo)
    liquidos_flags = [
        'flag_hemoderivados', 'flag_aferesis',
        'flag_perdidas_altas', 'flag_balance_extremo'
    ]
    df_result['flag_liquidos'] = (df_result[liquidos_flags].sum(axis=1) > 0).astype(int)
    
    print("\nVALIDACIÓN LÍQUIDOS:")
    for flag in liquidos_flags:
        print_validation_result(flag, df_result[flag], len(df_result))
    print(f"\n  TOTAL flag_liquidos: {df_result['flag_liquidos'].sum():,} ({df_result['flag_liquidos'].mean()*100:.2f}%)")
    
    return df_result
