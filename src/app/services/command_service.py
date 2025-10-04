from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
import uuid
from datetime import datetime, timezone

from app.models import Command as CommandModel, Turn as TurnModel
from app.schemas import (
    CommandCreate,
    CommandOut,
    CommandValidateIn,
    CommandCostIn,
    CommandIdOut,
)

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc).replace(microsecond=0)

def _to_iso_z(dt: datetime) -> str:
    # DB가 timezone-aware datetime을 저장한다고 가정
    # (만약 str로 저장 중이면 그대로 반환해도 무방)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            # 안전장치: naive면 UTC로 간주
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return str(dt)

def _new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _to_out(obj: CommandModel) -> CommandOut:
    return CommandOut(
        id=obj.id,
        turn_id=obj.turn_id,
        text=obj.text,
        validity=obj.validity,
        cost=obj.cost,
        created_at=_to_iso_z(obj.created_at),
    )

def create(db: Session, turn_id: str, body: CommandCreate) -> CommandIdOut:
    # Turn 존재 확인
    turn = db.get(TurnModel, turn_id)
    if not turn:
        raise ValueError("Turn not found")

    cid = _new_id("c")
    cmd = CommandModel(
        id=cid,
        turn_id=turn_id,
        text=body.text,
        created_at=_now_utc(),
    )
    db.add(cmd)
    db.commit()
    # create는 id만 반환 (사양 변경사항 반영)
    return CommandIdOut(command_id=cid)

def get(db: Session, cmd_id: str) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    return _to_out(obj) if obj else None

def list_by_turn(
    db: Session,
    turn_id: str,
    limit: int = 100,
    cursor: Optional[str] = None
) -> List[CommandOut]:
    q = (
        db.query(CommandModel)
        .filter(CommandModel.turn_id == turn_id)
        .order_by(CommandModel.id)
    )
    if cursor:
        q = q.filter(CommandModel.id > cursor)
    rows = q.limit(limit).all()
    return [_to_out(o) for o in rows]

def set_validity(db: Session, cmd_id: str, v: CommandValidateIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.validity = {
        "is_valid": v.is_valid,
        "score": v.score,
        "reasons": v.reasons or [],
    }
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

def set_cost(db: Session, cmd_id: str, c: CommandCostIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.cost = {
        "estimate": c.estimate,
        "currency": c.currency,
        "breakdown": c.breakdown or [],
    }
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)
"""
# app/services/command_service.py
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timezone

from app.models import Command as CommandModel, Turn as TurnModel
from app.schemas import CommandCreate, CommandOut, CommandValidateIn, CommandCostIn

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def to_out(obj: CommandModel) -> CommandOut:
    return CommandOut(
        id=obj.id,
        turn_id=obj.turn_id,
        text=obj.text,
        validity=obj.validity,
        cost=obj.cost,
        credated_at=obj.created_at,
    )

def create(db: Session, turn_id: str, body: CommandCreate) -> CommandOut:
    # turn 존재 확인
    turn = db.get(TurnModel, turn_id)
    if not turn:
        raise ValueError("Turn not found")

    cid = _new_id("c")
    cmd = CommandModel(
        id=cid,
        turn_id=turn_id,
        text=body.text,
        created_at=now_iso(),
    )
    db.add(cmd)
    db.commit()
    db.refresh(cmd)
    return to_out(cmd)

def get(db: Session, cmd_id: str) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    return to_out(obj) if obj else None

def list_by_turn(db: Session, turn_id: str, limit: int = 100, cursor: Optional[str] = None) -> List[CommandOut]:
    q = db.query(CommandModel).filter(CommandModel.turn_id == turn_id).order_by(CommandModel.id)
    if cursor:
        q = q.filter(CommandModel.id > cursor)
    q = q.limit(limit)
    return [to_out(o) for o in q.all()]

def set_validity(db: Session, cmd_id: str, v: CommandValidateIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.validity = {"is_valid": v.is_valid, "score": v.score, "reasons": v.reasons or []}
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return to_out(obj)

def set_cost(db: Session, cmd_id: str, c: CommandCostIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.cost = {"estimate": c.estimate, "currency": c.currency, "breakdown": c.breakdown or []}
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return to_out(obj)
"""