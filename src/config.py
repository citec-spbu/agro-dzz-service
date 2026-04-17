from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DEBUG: bool = False
    TITLE: str
    VERSION: str
    MAX_CLOUD: float = 30.0
    REFRESH_CHUNK_DAYS: int = 7
    DAYS_BACK: int = 45
    DAYS_FORWARD: int = 7
    CACHE_TTL_HOURS: int = 24
    GATEWAY_URL: str = "http://api-gateway:8080"
    FIELDS_INTERNAL_BASE_URL: str = "http://fields-service:8080"
    DATABASE_URL: str = "postgresql://dzz:dzz@db:5432/dzz"
    LEGACY_SQLITE_PATH: str | None = None
    PLANETARY_SYNC_ENABLED: bool = True
    PLANETARY_SYNC_HOUR_UTC: int = 2
    PLANETARY_SYNC_MINUTE_UTC: int = 0

    model_config = SettingsConfigDict(
        env_prefix="APP_", env_file=".env", extra="ignore"
    )


settings = Settings()
