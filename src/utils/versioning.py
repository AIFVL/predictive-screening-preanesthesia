from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_pipeline_hash(config_dir: Path | str, data_dir: Path | str) -> str:
    """Calcula SHA256 del contenido de todos los YAMLs + archivos de datos crudos."""
    config_dir = Path(config_dir)
    data_dir = Path(data_dir)

    h = hashlib.sha256()

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        h.update(yaml_file.name.encode())
        h.update(yaml_file.read_bytes())

    for data_file in sorted(data_dir.glob("*.xlsx")):
        h.update(data_file.name.encode())
        # Solo hash del tamaño + primeros 64KB para no leer archivos enormes completos
        stat = data_file.stat()
        h.update(str(stat.st_size).encode())
        with open(data_file, "rb") as f:
            h.update(f.read(65536))

    return h.hexdigest()


def resolve_version_dir(output_base: Path | str, version: str) -> Path:
    return Path(output_base) / version


def is_version_cached(version_dir: Path | str, config_dir: Path | str, data_dir: Path | str) -> bool:
    """Retorna True si ya existe un output para el hash actual."""
    version_dir = Path(version_dir)
    hash_file = version_dir / "pipeline_hash.json"

    if not hash_file.exists():
        return False

    try:
        stored = json.loads(hash_file.read_text())["hash"]
    except (KeyError, json.JSONDecodeError):
        return False

    current = compute_pipeline_hash(config_dir, data_dir)
    return stored == current


def save_pipeline_hash(version_dir: Path | str, config_dir: Path | str, data_dir: Path | str) -> str:
    """Calcula y guarda el hash en pipeline_hash.json. Retorna el hash."""
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    h = compute_pipeline_hash(config_dir, data_dir)
    hash_file = version_dir / "pipeline_hash.json"
    hash_file.write_text(json.dumps({"hash": h}, indent=2))
    return h
