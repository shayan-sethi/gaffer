import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        PROJECT_NAME: str = "Gaffer Sports Intelligence"
        ENVIRONMENT: str = "development"
        API_V1_STR: str = "/api"
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        DEBUG: bool = True

        POSTGRES_SERVER: str = "localhost"
        POSTGRES_PORT: int = 5432
        POSTGRES_USER: str = "gaffer"
        POSTGRES_PASSWORD: str = "gaffer_dev_password"
        POSTGRES_DB: str = "gaffer_db"
        DATABASE_URL: Optional[str] = None

        REDIS_HOST: str = "localhost"
        REDIS_PORT: int = 6379
        REDIS_URL: Optional[str] = None
        CELERY_BROKER_URL: Optional[str] = None
        CELERY_RESULT_BACKEND: Optional[str] = None

        S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
        S3_ACCESS_KEY: str = "minioadmin"
        S3_SECRET_KEY: str = "minioadmin"
        S3_BUCKET_NAME: str = "gaffer-matches"
        S3_REGION: str = "us-east-1"
        S3_USE_SSL: bool = False

        YOLO_MODEL_PATH: str = "yolov8n.pt"
        CONFIDENCE_THRESHOLD: float = 0.35
        IOU_THRESHOLD: float = 0.45
        STORAGE_LOCAL_DIR: str = "/tmp/gaffer/storage"

        @property
        def sync_database_url(self) -> str:
            if self.DATABASE_URL:
                return self.DATABASE_URL
            return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

        @property
        def broker_url(self) -> str:
            if self.CELERY_BROKER_URL:
                return self.CELERY_BROKER_URL
            if self.REDIS_URL:
                return self.REDIS_URL
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

        @property
        def result_backend(self) -> str:
            if self.CELERY_RESULT_BACKEND:
                return self.CELERY_RESULT_BACKEND
            return self.broker_url

    settings = Settings()

except Exception:
    try:
        from pydantic import BaseSettings
        class Settings(BaseSettings):
            PROJECT_NAME: str = "Gaffer Sports Intelligence"
            ENVIRONMENT: str = "development"
            API_V1_STR: str = "/api"
            HOST: str = "0.0.0.0"
            PORT: int = 8000
            DEBUG: bool = True

            POSTGRES_SERVER: str = "localhost"
            POSTGRES_PORT: int = 5432
            POSTGRES_USER: str = "gaffer"
            POSTGRES_PASSWORD: str = "gaffer_dev_password"
            POSTGRES_DB: str = "gaffer_db"
            DATABASE_URL: Optional[str] = None

            REDIS_HOST: str = "localhost"
            REDIS_PORT: int = 6379
            REDIS_URL: Optional[str] = None
            CELERY_BROKER_URL: Optional[str] = None
            CELERY_RESULT_BACKEND: Optional[str] = None

            S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
            S3_ACCESS_KEY: str = "minioadmin"
            S3_SECRET_KEY: str = "minioadmin"
            S3_BUCKET_NAME: str = "gaffer-matches"
            S3_REGION: str = "us-east-1"
            S3_USE_SSL: bool = False

            YOLO_MODEL_PATH: str = "yolov8n.pt"
            CONFIDENCE_THRESHOLD: float = 0.35
            IOU_THRESHOLD: float = 0.45
            STORAGE_LOCAL_DIR: str = "/tmp/gaffer/storage"

            @property
            def sync_database_url(self) -> str:
                if self.DATABASE_URL:
                    return self.DATABASE_URL
                return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

            @property
            def broker_url(self) -> str:
                if self.CELERY_BROKER_URL:
                    return self.CELERY_BROKER_URL
                if self.REDIS_URL:
                    return self.REDIS_URL
                return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

            @property
            def result_backend(self) -> str:
                if self.CELERY_RESULT_BACKEND:
                    return self.CELERY_RESULT_BACKEND
                return self.broker_url

            class Config:
                env_file = ".env"
                extra = "ignore"

        settings = Settings()
    except Exception:
        # Pure Python fallback
        class PurePythonSettings:
            PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Gaffer Sports Intelligence")
            ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
            API_V1_STR: str = os.getenv("API_V1_STR", "/api")
            HOST: str = os.getenv("HOST", "0.0.0.0")
            PORT: int = int(os.getenv("PORT", "8000"))
            DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

            POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
            POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
            POSTGRES_USER: str = os.getenv("POSTGRES_USER", "gaffer")
            POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "gaffer_dev_password")
            POSTGRES_DB: str = os.getenv("POSTGRES_DB", "gaffer_db")
            DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

            REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
            REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
            REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
            CELERY_BROKER_URL: Optional[str] = os.getenv("CELERY_BROKER_URL")
            CELERY_RESULT_BACKEND: Optional[str] = os.getenv("CELERY_RESULT_BACKEND")

            S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
            S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
            S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
            S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "gaffer-matches")
            S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
            S3_USE_SSL: bool = os.getenv("S3_USE_SSL", "False").lower() in ("true", "1")

            YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
            CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
            IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
            STORAGE_LOCAL_DIR: str = os.getenv("STORAGE_LOCAL_DIR", "/tmp/gaffer/storage")

            @property
            def sync_database_url(self) -> str:
                if self.DATABASE_URL:
                    return self.DATABASE_URL
                return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

            @property
            def broker_url(self) -> str:
                if self.CELERY_BROKER_URL:
                    return self.CELERY_BROKER_URL
                if self.REDIS_URL:
                    return self.REDIS_URL
                return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

            @property
            def result_backend(self) -> str:
                if self.CELERY_RESULT_BACKEND:
                    return self.CELERY_RESULT_BACKEND
                return self.broker_url

        settings = PurePythonSettings()

os.makedirs(settings.STORAGE_LOCAL_DIR, exist_ok=True)
