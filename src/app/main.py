from fastapi import FastAPI
from app.api.v1.endpoints import health

app = FastAPI(title="[NSAC] Screwed-Backend API", version="0.1.0")

app.include_router(health.router, prefix="/v1", tags=["health"])
# app.include_router(chat.router,   prefix="/v1", tags=["chat"])
# app.include_router(predict.router, prefix="/v1", tags=["predict"])

# 실행: uvicorn app.main:app --reload