from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Preanesthesia Screening API"
    app_version: str = "1.0.0"
    description: str = (
        "Generic REST API for uploading and serving preanesthesia screening ML models. "
        "Adapts dynamically to any feature set stored in model artifacts."
    )

    models_dir: Path = Path("models")

    max_batch_size: int = 100
    enable_shap: bool = True
    shap_background_samples: int = 100

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_config = {"env_file": ".env", "env_prefix": "API_"}


settings = Settings()
