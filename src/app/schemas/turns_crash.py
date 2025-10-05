# app/schemas/turns_crash.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal

class CrashNodeInput(BaseModel):
    current_id: str = Field(..., description="클라이언트가 현재로 인식하는 턴 ID")
    stats: Dict[str, Any] = Field(default_factory=dict, description="이 턴에서 기록할 통계")
    resolution: Optional[Literal["FORK"]] = Field(
        None, description="충돌 시 강제로 포크하려면 'FORK' 지정"
    )

class ConflictOut(BaseModel):
    error: Literal["TIMELINE_CONFLICT"] = "TIMELINE_CONFLICT"
    message: str
    latest_id: str
    latest_etag: str
    resolution: Literal["FORK"] = "FORK"
    branch_id: str