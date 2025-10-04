from fastapi import FastAPI
from app.core.config import settings
from app.api.v1 import api_router as api_v1_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=None,
)

app.include_router(api_v1_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health():
    return {"status": "ok"}