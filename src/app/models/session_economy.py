# app/models/session_economy.py
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from app.db.base import Base

def now_dt():
    return datetime.now(tz=timezone.utc)

class SessionEconomy(Base):
    __tablename__ = "session_economy"

    id = Column(String, primary_key=True, default=lambda: f"eco_{uuid.uuid4().hex[:8]}")
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    currency = Column(String(3), nullable=False, default="USD")
    balance = Column(Numeric(18, 2), nullable=False, default=0)  # 현재 잔액
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now_dt, onupdate=now_dt)

    # Relationship
    session = relationship("Session", back_populates="economy")
