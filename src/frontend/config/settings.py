
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_url: str

    model_config = SettingsConfigDict(
        extra="allow",
    )
