from fastapi import FastAPI
# app/main.py
from app.core.config import settings
print(">>> DATABASE_URL:", settings.DATABASE_URL)
from app.api.v1 import api_router as api_v1_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=None,
)

app.include_router(api_v1_router, prefix=settings.API_V1_STR)