import pandas as pd

# Mapeo de nombres: variables_seleccionadas.csv → columnas reales de OPERA_BASE_clean.csv
SEVERITY_SCORE_MAP = {
    'score_proc_high_severity': 'severity_score_high_severity_proc',
    'score_proc_critical': 'severity_score_critical_proc',
    'score_proc_moderate_severity': 'severity_score_moderate_severity_proc',
    'score_proc_medium_severity': 'severity_score_medium_severity_proc',
    'score_proc_low_severity': 'severity_score_low_severity_proc',
    'score_dx_low_severity': 'severity_score_low_severity_dx',
    'score_dx_high_severity': 'severity_score_high_severity_dx',
    'score_dx_critical': 'severity_score_critical_dx',
    'score_dx_medium_severity': 'severity_score_medium_severity_dx',
    'score_dx_moderate_severity': 'severity_score_moderate_severity_dx',
    'score_proc_ant_medium_severity': 'severity_score_medium_severity_proc_ant',
    'score_proc_ant_moderate_severity': 'severity_score_moderate_severity_proc_ant',
    'score_proc_ant_critical': 'severity_score_critical_proc_ant',
    'score_proc_ant_low_severity': 'severity_score_low_severity_proc_ant',
    'score_proc_ant_high_severity': 'severity_score_high_severity_proc_ant',
}

LABEL_ENCODED_MAP = {
    'predicted_label_proc_encoded': 'severity_ordinal_proc',
    'predicted_label_dx_encoded': 'severity_ordinal_dx',
    'predicted_label_proc_ant_encoded': 'severity_ordinal_proc_ant',
}

ENCODING_FIX_MAP = {
    'Prótesis Dental_movil': 'Prótesis Dental_movil',
    'Prótesis Dental_no': 'Prótesis Dental_no',
    'Alérgeno_med_opioides': 'Alérgeno_med_opioides',
    'Alérgeno_suplementos': 'Alérgeno_suplementos',
}

def resolve_feature_columns(selected_names, df_columns, features_meta):
    """
    Mapea nombres de features seleccionadas a columnas reales del DataFrame.
    
    Args:
        selected_names (list): Lista de nombres de features (desde variables_seleccionadas.csv).
        df_columns (list): Lista de columnas disponibles en el DataFrame principal.
        features_meta (pd.DataFrame): DataFrame con metadatos (Variable, Type, etc.).
        
    Returns:
        resolved_cols (list): Columnas del df a usar como features.
        missing (list): Nombres no resueltos.
    """
    df_cols_set = set(df_columns)
    resolved = []
    missing = []
    
    # Obtener tipos de variable del metadata
    type_map = dict(zip(features_meta['Variable'], features_meta['Type']))
    
    for name in selected_names:
        var_type = type_map.get(name, '')
        
        # 1. OneHot_Group: buscar todas las columnas con ese prefijo
        if var_type == 'OneHot_Group':
            prefix = name + '_'
            matching = [c for c in df_columns if c.startswith(prefix)]
            if matching:
                resolved.extend(matching)
            else:
                missing.append(f"{name} (OneHot_Group, no prefix match)")
        
        # 2. Scores de severidad NLP
        elif name in SEVERITY_SCORE_MAP:
            real_col = SEVERITY_SCORE_MAP[name]
            if real_col in df_cols_set:
                resolved.append(real_col)
            else:
                missing.append(f"{name} → {real_col} (not found)")
        
        # 3. Labels encoded
        elif name in LABEL_ENCODED_MAP:
            real_col = LABEL_ENCODED_MAP[name]
            if real_col in df_cols_set:
                resolved.append(real_col)
            else:
                missing.append(f"{name} → {real_col} (not found)")
        
        # 4. Match directo
        elif name in df_cols_set:
            resolved.append(name)
        
        # 5. Encoding fix: buscar con/sin tildes, ñ, etc.
        else:
            # Intentar match parcial (sin acentos problemáticos)
            found = False
            for col in df_columns:
                if name.replace('Ã³', 'ó').replace('Ã©', 'é') == col or \
                   col.replace('ó', 'Ã³').replace('é', 'Ã©') == name:
                    resolved.append(col)
                    found = True
                    break
            if not found:
                # Buscar por similitud de inicio
                for col in df_columns:
                    if col.startswith(name[:10]) and len(col) == len(name):
                        resolved.append(col)
                        found = True
                        break
            if not found:
                missing.append(f"{name} (no match)")
    
    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_resolved = []
    for col in resolved:
        if col not in seen:
            seen.add(col)
            unique_resolved.append(col)
    
    return unique_resolved, missing
