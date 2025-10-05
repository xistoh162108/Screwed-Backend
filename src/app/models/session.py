from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone
from app.db.base import Base  # 프로젝트의 Base import 경로에 맞추세요

# 기존
def now_iso():
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# 변경 (datetime 객체 반환)
def now_dt():
    return datetime.now(tz=timezone.utc)
class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)  # 예: s_ab12cd34
    title = Column(String, nullable=True)
    # 선택: 세션의 루트 Turn을 연결하고 싶다면 FK
    root_turn_id = Column(String, ForeignKey("turns.id"), nullable=True)

    meta = Column(JSONB, nullable=True)  # 임의 메타데이터
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now(), nullable=False)