from fastapi import APIRouter
from app.api.v1.turns import router as turns_router

api_router = APIRouter()
api_router.include_router(turns_router)