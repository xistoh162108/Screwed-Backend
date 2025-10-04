# app/models/command.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import Column, String, ForeignKey, JSON as SAJSON
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.core.config import settings

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql")

JSONType = JSONB if is_postgres_url(settings.DATABASE_URL) else SAJSON().with_variant(SQLITE_JSON, "sqlite")

class Command(Base):
    __tablename__ = "commands"

    id = Column(String, primary_key=True, index=True)         # 예: c_ab12cd34
    turn_id = Column(String, ForeignKey("turns.id"), index=True, nullable=False)

    text = Column(String, nullable=False)                      # 자연어 지시
    validity = Column(JSONType, nullable=True)                 # {is_valid, score, reasons[]}
    cost = Column(JSONType, nullable=True)                     # {estimate, currency, breakdown[]}

    created_at = Column(String, nullable=False, default=now_iso)

    # 관계(옵션): 필요 시 역참조 사용
    turn = relationship("Turn", backref="commands_rel", lazy="joined")