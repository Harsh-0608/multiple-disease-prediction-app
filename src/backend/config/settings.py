from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    PROJECT_ROOT: Path

    LOG_PATH: Path

    heart_disease_model_path: Path
    diabetes_model_path: Path


    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    def get_path(self, path: Path) -> Path:
        return self.PROJECT_ROOT / path

    @property
    def log_path_absolute(self) -> Path:
        return self.get_path(self.LOG_PATH)

    @property
    def diabetes_model_path_absolute(self) -> Path:
        return self.get_path(self.diabetes_model_path)

    @property
    def heart_disease_model_path_absolute(self) -> Path:
        return self.get_path(self.heart_disease_model_path)

    # @property
    # def diabetes_dataset_path_absolute(self) -> Path:
    #     return self.get_path(self.diabetes_dataset_path)

    # @property
    # def heart_disease_dataset_path_absolute(self) -> Path:
    #     return self.get_path(self.heart_disease_dataset_path)

    # @property
    # def hyper_params_yaml_path_absolute(self) -> Path:
    #     return self.get_path(self.hyper_params_yaml_path)

        