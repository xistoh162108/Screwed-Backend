from sqlalchemy import Column, String, Enum, JSON, DateTime
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON  # 호환용
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from datetime import datetime, timezone
import enum

from app.db.base import Base

def now_iso():
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

class TurnState(str, enum.Enum):
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

# SQLite JSON 타입 대비
JSONType = JSON().with_variant(SQLITE_JSON, "sqlite")

class Turn(Base):
    __tablename__ = "turns"

    id = Column(String, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("turns.id"), nullable=True)
    branch_id = Column(String, index=True, nullable=True)
    month = Column(String, index=True, nullable=False)         # "YYYY-MM"
    state = Column(Enum(TurnState), index=True, nullable=False, default=TurnState.DRAFT)
    stats = Column(JSONType, nullable=False, default=dict)     # 그대로 JSON 저장
    created_at = Column(String, nullable=False, default=now_iso)
    updated_at = Column(String, nullable=False, default=now_iso)

    parent = relationship("Turn", remote_side=[id], backref="children_rel", lazy="joined")