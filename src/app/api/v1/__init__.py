from fastapi import APIRouter
from app.api.v1.turns import router as turns_router
from app.api.v1.commands import router as commands_router   # 추가
# (health 라우터가 있으면 같이 include)

api_router = APIRouter()
api_router.include_router(turns_router)
api_router.include_router(commands_router)  # 추가