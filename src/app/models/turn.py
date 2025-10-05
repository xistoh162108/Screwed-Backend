from __future__ import annotations
import enum
from datetime import datetime, timezone

from sqlalchemy import Column, String, ForeignKey, func, DateTime
from sqlalchemy.types import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base
# (참고) 문자열이 아니라 datetime 객체가 필요하면 이렇게 사용하세요.
def now_dt() -> datetime:
    return datetime.now(tz=timezone.utc)

JSONType = JSONB

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


class Turn(Base):
    __tablename__ = "turns"

    id = Column(String, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("turns.id"), nullable=True)
    branch_id = Column(String, index=True, nullable=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    month = Column(String, index=True, nullable=False)  # "YYYY-MM"

    state = Column(
        SQLEnum(TurnState, name="turn_state", native_enum=True),  # name을 주는게 PG에서 안전
        index=True,
        nullable=False,
        default=TurnState.DRAFT,
    )

    stats = Column(JSONType, nullable=False, default=dict)

    # ✅ 문자열이 아니라 타임스탬프 컬럼로 변경
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    parent = relationship(
        "Turn",
        remote_side=[id],
        backref="children_rel",
        lazy="joined",
    )