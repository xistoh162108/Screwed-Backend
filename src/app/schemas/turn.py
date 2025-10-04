from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

class TurnState(str, Enum):
    DRAFT = "DRAFT"
    BRIEFED = "BRIEFED"
    COMMAND_PENDING = "COMMAND_PENDING"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    COST_ESTIMATED = "COST_ESTIMATED"
    BUDGET_OK = "BUDGET_OK"
    REJECTED = "REJECTED"
    SIMULATED = "SIMULATED"
    APPLIED = "APPLIED"

# Stats는 자유로운 JSON 구조를 허용(간단화)
class Stats(BaseModel):
    climate: Optional[Dict[str, Any]] = None
    yield_: Optional[Dict[str, Any]] = Field(None, alias="yield")
    env: Optional[Dict[str, Any]] = None
    money: Optional[Dict[str, Any]] = None
    notes: Optional[List[str]] = None

    class Config:
        populate_by_name = True  # "yield"로 입출력 가능

class TurnBase(BaseModel):
    parent_id: Optional[str] = None
    branch_id: Optional[str] = None
    month: str
    
    @field_validator("parent_id", "branch_id", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

class TurnCreate(BaseModel):
    parent_id: Optional[str] = None
    branch_id: Optional[str] = None

class TurnUpdateStats(BaseModel):
    stats: Stats

class TurnOut(TurnBase):
    id: str
    children: List[str] = []
    state: TurnState
    stats: Stats = Stats()
    created_at: str
    updated_at: str