# app/schemas/command.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CommandCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)

class CommandIdOut(BaseModel):
    command_id: str

class CommandOut(BaseModel):
    id: str
    turn_id: str
    text: str
    validity: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None
    created_at: str
    class Config:
        from_attributes = True

class CommandValidateIn(BaseModel):
    is_valid: bool
    score: Optional[float] = None
    reasons: Optional[List[str]] = None

class CommandCostIn(BaseModel):
    estimate: float
    currency: str = "KRW"
    breakdown: Optional[List[Dict[str, Any]]] = None