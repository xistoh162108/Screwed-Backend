# app/models/turn.py
from __future__ import annotations

from datetime import datetime, timezone
import enum

from sqlalchemy import Column, String, Enum as SAEnum, ForeignKey, JSON as SAJSON
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.core.config import settings


def now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_postgres_url(url: str) -> bool:
    # postgresql / postgresql+psycopg / postgresql+pg8000 ... 모두 매칭
    return url.startswith("postgresql")


# ---------------------------------------------------------------------
# JSON 타입 멀티백엔드 스위치
#  - Postgres → JSONB
#  - 그 외(SQLite 등) → 일반 JSON(+sqlite 변형)
# ---------------------------------------------------------------------
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
    month = Column(String, index=True, nullable=False)  # "YYYY-MM"

    # enum은 python enum과 매칭되도록 SQLAlchemy Enum 사용
    state = Column(
        SAEnum(TurnState),
        index=True,
        nullable=False,
        default=TurnState.DRAFT,
    )

    # 멀티백엔드 JSON 타입
    stats = Column(JSONType, nullable=False, default=dict)

    created_at = Column(String, nullable=False, default=now_iso)
    updated_at = Column(String, nullable=False, default=now_iso)

    # 자기참조 관계 (부모/자식)
    parent = relationship(
        "Turn",
        remote_side=[id],
        backref="children_rel",
        lazy="joined",
    )