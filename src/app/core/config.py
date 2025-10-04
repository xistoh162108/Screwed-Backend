from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    PROJECT_NAME: str = "[NSAC] Backend API"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "postgresql+psycopg://xistoh:mk685700@34.64.87.169:5432/screwed"
    class Config: env_file = ".env"
    # 예: "postgresql+psycopg://user:pass@host:5432/dbname"

settings = Settings()