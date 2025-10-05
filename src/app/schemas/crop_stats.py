# app/schemas/crop_stats.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal
from decimal import Decimal

Crop = Literal["MAIZE", "RICE", "SOYBEAN", "WHEAT"]

class SessionCropStatsBase(BaseModel):
    crop: Crop
    cumulative_production_tonnes: Decimal = Field(default=0, max_digits=18, decimal_places=3)
    co2e_kg: Decimal = Field(default=0, max_digits=18, decimal_places=3)
    water_m3: Decimal = Field(default=0, max_digits=18, decimal_places=3)
    fert_n_kg: Decimal = Field(default=0, max_digits=18, decimal_places=3)
    fert_p_kg: Decimal = Field(default=0, max_digits=18, decimal_places=3)
    fert_k_kg: Decimal = Field(default=0, max_digits=18, decimal_places=3)

class SessionCropStatsCreate(SessionCropStatsBase):
    pass

class SessionCropStatsUpsert(SessionCropStatsBase):
    """PATCH/PUT에서 누적 값을 덮어쓰거나(absolute) 더할지(delta) 선택"""
    mode: Literal["absolute", "delta"] = "absolute"

class SessionCropStatsOut(SessionCropStatsBase):
    id: str
    session_id: str
    last_event_at: datetime

    class Config:
        from_attributes = True

class SessionCropStatsListOut(BaseModel):
    items: List[SessionCropStatsOut]
