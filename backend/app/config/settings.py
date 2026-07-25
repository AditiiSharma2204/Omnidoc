from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "OmniDoc AI"
    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api/v1"

    UPLOAD_DIR: str = "storage/uploads"
    PARSED_DIR: str = "storage/parsed"
    VECTORSTORE_DIR: str = "vectorstore"

    class Config:
        env_file = ".env"


settings = Settings()