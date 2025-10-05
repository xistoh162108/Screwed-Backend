# app/schemas/simulation.py
"""
시뮬레이션 API 스키마
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal


# ============================================
# Request Schemas
# ============================================

class SimulationInput(BaseModel):
    """시뮬레이션 입력"""
    normalized_text: str = Field(..., min_length=1, max_length=500)
    action_count: int = Field(..., ge=0, le=10)
    seed: int = Field(default=42, ge=0)
    window: int = Field(default=6, ge=3, le=30)
    force_recompute: bool = Field(default=False)


# ============================================
# Response Schemas
# ============================================

class WindowInfo(BaseModel):
    """윈도우 정보"""
    start_row: int
    end_row: int
    seed: int


class DeltaInfo(BaseModel):
    """델타 정보"""
    op: Literal["*", "+"]
    value: float


class WeatherPrediction(BaseModel):
    """기상 예측"""
    T2M_next: Optional[float] = None
    PRECTOTCORR_next: Optional[float] = None
    RH2M_next: Optional[float] = None
    # ... 필요한 변수 추가


class YieldPrediction(BaseModel):
    """생산량 예측"""
    crop: Optional[str] = None
    area_acre: float = 0
    area_ha: float = 0
    yield_ton_pred: float = 0


class Predictions(BaseModel):
    """예측 결과"""
    weather: Dict[str, Any] = {}
    yield_: YieldPrediction = Field(default_factory=YieldPrediction, alias="yield")

    class Config:
        populate_by_name = True


class LLMFeedback(BaseModel):
    """LLM 피드백"""
    summary: str
    reasons: List[str] = []
    actions_next: List[str] = []
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class AuditInfo(BaseModel):
    """감사 정보"""
    session_id: str
    turn_id: str
    model_versions: Dict[str, str] = {}
    prompt_hash: Optional[str] = None
    data_snapshot_id: Optional[str] = None
    reused_cache: bool = False


class SimulationOutput(BaseModel):
    """시뮬레이션 출력 (정상)"""
    ok: bool = True
    violation: Optional[str] = None
    applied_window: WindowInfo
    applied_deltas: Dict[str, DeltaInfo]
    predictions: Predictions
    llm_feedback: Optional[LLMFeedback] = None
    audit: AuditInfo


class SimulationViolation(BaseModel):
    """시뮬레이션 출력 (위반)"""
    ok: bool = False
    violation: str
    error_code: str
