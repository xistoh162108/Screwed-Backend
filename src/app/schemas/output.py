from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field

# 생성 시 선택적으로 모델 지정 등 메타만 받음
class OutputCreateIn(BaseModel):
    # 선택: 어떤 모델로 처리할지 지정(미지정 시 서버 기본값)
    fine_tuned_model_id: Optional[str] = None
    foundation_model_id: Optional[str] = None
    # 기타 파이프라인 옵션
    options: Optional[Dict[str, Any]] = None

class OutputIdOut(BaseModel):
    output_id: str

class OutputOut(BaseModel):
    id: str
    command_id: str
    turn_id: str
    state: Literal["PENDING", "RUNNING", "COMPLETE", "FAILED"]
    kind: Optional[Literal["DENIED", "ANSWER", "PROCESSED"]] = None
    answer: Optional[str] = None
    impact: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    delta_stats: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    assumptions: Optional[List[str]] = None
    denied_reasons: Optional[List[str]] = None
    created_at: str
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True  # pydantic v2