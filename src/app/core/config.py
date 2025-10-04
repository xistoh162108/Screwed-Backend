# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # ===== 기본 정보 =====
    PROJECT_NAME: str = "[NSAC] Backend API"
    API_V1_STR: str = "/api/v1"
    API_VERSION: str = "0.1.2"

    # ===== DB 설정 =====
    # 기본은 DATABASE_URL을 사용. (docker-compose나 .env에서 주입)
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://xistoh:mk685700@34.64.87.169:5432/screwed",
        alias="DATABASE_URL",
    )
    # 혹시 기존 키를 쓰고 있다면 fallback 허용 (선택)
    APP_SQLALCHEMY_URI: str | None = Field(default=None, alias="APP_SQLALCHEMY_URI")

    # ===== Celery/Redis =====
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/0", alias="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/0", alias="CELERY_RESULT_BACKEND")

    # ===== Pydantic v2 설정 (class Config 금지) =====
    model_config = SettingsConfigDict(
        env_file=".env",       # .env 읽기
        extra="ignore",        # 정의 안 된 환경변수 무시 (이번 에러 방지 포인트)
        case_sensitive=False,  # 대소문자 구분 안 함
    )

settings = Settings()

# APP_SQLALCHEMY_URI가 주어졌고, DATABASE_URL이 비어있다면 fallback (선택)
if settings.APP_SQLALCHEMY_URI and not settings.DATABASE_URL:
    settings.DATABASE_URL = settings.APP_SQLALCHEMY_URI