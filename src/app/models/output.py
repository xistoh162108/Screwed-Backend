from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy import JSON as SAJSON
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.core.config import settings

# ----- 공통 유틸 -----
def now_iso() -> str:
    # ISO-8601 UTC, 'Z' 접미사
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

def is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql")

# Postgres면 JSONB, 아니면 각 엔진별 JSON
JSONType = JSONB if is_postgres_url(settings.DATABASE_URL) else SAJSON().with_variant(SQLITE_JSON, "sqlite")

class Output(Base):
    """
    Command 하나당 정확히 하나의 Output.
    상태(state): PENDING | RUNNING | COMPLETE | FAILED
    종류(kind):  DENIED  | ANSWER  | PROCESSED
    """
    __tablename__ = "outputs"

    # 식별자
    id = Column(String, primary_key=True, index=True)                   # 예: o_12ab34cd
    command_id = Column(String, ForeignKey("commands.id"), nullable=False, unique=True)
    turn_id = Column(String, ForeignKey("turns.id"), nullable=False, index=True)

    # 상태/종류
    state = Column(String, nullable=False)                              # PENDING | RUNNING | COMPLETE | FAILED
    kind = Column(String, nullable=True)                                # DENIED | ANSWER | PROCESSED

    # 결과 페이로드
    answer = Column(String, nullable=True)                              # kind=ANSWER일 때만 사용
    impact = Column(JSONType, nullable=True)                            # { env_delta, emission, ... }
    prediction = Column(JSONType, nullable=True)                        # { next_month_stats, ... }
    delta_stats = Column(JSONType, nullable=True)                       # 적용 시 반영 요약
    models = Column(JSONType, nullable=True)                            # 사용 모델 메타
    assumptions = Column(JSONType, nullable=True)                       # 가정 리스트
    denied_reasons = Column(JSONType, nullable=True)                    # DENIED 사유

    # 타임스탬프(문자열 ISO-8601, Command와 통일)
    created_at = Column(String, nullable=False, default=now_iso)
    completed_at = Column(String, nullable=True)

    # 관계(옵션)
    command = relationship("Command", backref="output_rel", lazy="joined")
    turn = relationship("Turn", backref="outputs_rel", lazy="joined")

    __table_args__ = (
        UniqueConstraint("command_id", name="uq_outputs_command_id"),
        # Index("ix_outputs_turn_id", "turn_id"),
    )

    def __repr__(self) -> str:
        return f"<Output id={self.id} cmd={self.command_id} state={self.state} kind={self.kind}>"