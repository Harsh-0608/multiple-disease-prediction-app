
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    PROJECT_ROOT: Path

    LOG_PATH: Path

    heart_disease_model_path: Path
    diabetes_model_path: Path

    diabetes_target_col: str
    heart_disease_target_col: str

    RANDOM_STATE: int
    test_size: float

    hyper_params_yaml_path: Path

    heart_disease_dataset_path: Path
    diabetes_dataset_path: Path

    api_url: str

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()
