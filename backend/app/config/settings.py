from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "OmniDoc AI"
    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api/v1"

    DOCUMENTS_DIR: str = "storage/documents"
    VECTORSTORE_DIR: str = "vectorstore"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()