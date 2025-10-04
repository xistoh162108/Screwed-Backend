from pydantic import BaseModel
import os

class Settings(BaseModel):
    PROJECT_NAME: str = "[NSAC] Turn Management API"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    # 예: "postgresql+psycopg://user:pass@host:5432/dbname"

settings = Settings()