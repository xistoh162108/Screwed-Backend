# app/models/session_crop_stats.py
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.types import Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.db.base import Base
import enum

def now_dt():
    return datetime.now(tz=timezone.utc)

class Crop(str, enum.Enum):
    MAIZE = "MAIZE"
    RICE = "RICE"
    SOYBEAN = "SOYBEAN"
    WHEAT = "WHEAT"

class SessionCropStats(Base):
    __tablename__ = "session_crop_stats"

    id = Column(String, primary_key=True, default=lambda: f"crop_{uuid.uuid4().hex[:8]}")
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    crop = Column(SQLEnum(Crop, name="crop_enum", native_enum=True), nullable=False)

    # 누적 생산량 (톤 기준)
    cumulative_production_tonnes = Column(Numeric(18, 3), nullable=False, default=0)

    # 지속가능성 지표(누적)
    co2e_kg = Column(Numeric(18, 3), nullable=False, default=0)      # 탄소배출량
    water_m3 = Column(Numeric(18, 3), nullable=False, default=0)     # 물 사용량
    fert_n_kg = Column(Numeric(18, 3), nullable=False, default=0)    # 질소
    fert_p_kg = Column(Numeric(18, 3), nullable=False, default=0)    # 인
    fert_k_kg = Column(Numeric(18, 3), nullable=False, default=0)    # 칼륨

    last_event_at = Column(DateTime(timezone=True), nullable=False, default=now_dt, onupdate=now_dt)

    __table_args__ = (
        UniqueConstraint("session_id", "crop", name="uq_session_crop"),
    )

    # Relationship
    session = relationship("Session", back_populates="crop_stats")
