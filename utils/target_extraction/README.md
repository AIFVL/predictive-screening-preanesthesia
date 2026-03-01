# Target extraction module

Este paquete contiene la lógica de extracción de variable objetivo que antes estaba inline en el notebook `4_extract_target_variable.ipynb`.

## Estructura

- `constants.py`: listas de flags relevantes/excluidos.
- `helpers.py`: utilidades transversales.
- `specific`: carpeta de validadores específicos
- `target_builder.py`: construcción de `target` por umbral.
- `pipeline.py`: orquestación completa y exportación de resultados.

## Uso rápido

```python
from utils.target_extraction import run_target_extraction_pipeline

# df_postqx cargado previamente
# retorna: df_validated, df_v1, df_v2
df, df_v1, df_v2 = run_target_extraction_pipeline(df_postqx)
```

## Uso flexible (N versiones)

```python
from utils.target_extraction import run_target_extraction_pipeline, RELEVANT_FLAGS

versions = [
	{
		"name": "v1",
		"threshold": 1,
		"flags_to_use": RELEVANT_FLAGS,
		"export_name": "OPERA_POS_v1.xlsx",
		"description": "Base",
	},
	{
		"name": "v2",
		"threshold": 2,
		"flags_to_use": RELEVANT_FLAGS,
		"export_name": "OPERA_POS_v2.xlsx",
		"description": "Más estricta",
	},
	{
		"name": "v3_sin_liquidos",
		"threshold": 2,
		"flags_to_use": [f for f in RELEVANT_FLAGS if f != "flag_liquidos"],
		"export_name": "OPERA_POS_v3_sin_liquidos.xlsx",
		"description": "Excluye líquidos",
	},
]

# retorna: df_validated, dict de versiones {'v1': df, 'v2': df, ...}
df, df_versions = run_target_extraction_pipeline(df_postqx, versions=versions)
```
