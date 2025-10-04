from fastapi import APIRouter
from app.api.v1.turns import router as turns_router
from app.api.v1.commands import router as commands_router   # 추가
from app.api.v1.outputs import router as outputs_router   # 추가
from app.api.v1.health import router as health_router
from app.api.v1.sessions import router as sessions_router   # 추가

api_router = APIRouter()
api_router.include_router(turns_router)
api_router.include_router(commands_router)  # 추가
api_router.include_router(outputs_router)  # 추가
api_router.include_router(health_router)  # 추가
api_router.include_router(sessions_router)  # 추가