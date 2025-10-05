# app/services/output_repo.py
from typing import Optional, List
from sqlalchemy.orm import Session
import uuid
from datetime import datetime, timezone

from app.models import Output as OutputModel, Command as CommandModel, Turn as TurnModel
from app.schemas.output import OutputIdOut, OutputOut

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _normalize_iso(dt_val):
    if dt_val is None:
        return None
    if isinstance(dt_val, str):
        return dt_val.replace("+00:00", "Z")
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=timezone.utc)
    return dt_val.isoformat().replace("+00:00", "Z")

def _new_id(prefix: str = "o") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _to_out(obj: OutputModel) -> OutputOut:
    return OutputOut(
        id=obj.id,
        command_id=obj.command_id,
        turn_id=obj.turn_id,
        state=obj.state,
        kind=obj.kind,
        answer=obj.answer,
        impact=obj.impact,
        prediction=obj.prediction,
        delta_stats=obj.delta_stats,
        models=obj.models,
        assumptions=obj.assumptions,
        denied_reasons=obj.denied_reasons,
        created_at=_normalize_iso(obj.created_at),
        completed_at=_normalize_iso(obj.completed_at),
    )

# ---- 생성(멱등) ----
def create_from_command(db: Session, command_id: str) -> OutputIdOut:
    cmd = db.get(CommandModel, command_id)
    if not cmd:
        raise ValueError("Command not found")
    existing = db.query(OutputModel).filter(OutputModel.command_id == command_id).first()
    if existing:
        return OutputIdOut(output_id=existing.id)
    oid = _new_id("o")
    out = OutputModel(
        id=oid, command_id=command_id, turn_id=cmd.turn_id,
        state="PENDING", kind=None, answer=None,
        impact=None, prediction=None, delta_stats=None,
        models=None, assumptions=None, denied_reasons=None,
        created_at=_now_iso(), completed_at=None,
    )
    db.add(out)
    db.commit()
    return OutputIdOut(output_id=oid)

# ---- 조회 ----
def get_output(db: Session, output_id: str) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    return _to_out(obj) if obj else None

def list_by_turn(db: Session, turn_id: str, limit: int = 100, cursor: Optional[str] = None) -> List[OutputOut]:
    q = db.query(OutputModel).filter(OutputModel.turn_id == turn_id).order_by(OutputModel.id)
    if cursor:
        q = q.filter(OutputModel.id > cursor)
    return [_to_out(o) for o in q.limit(limit).all()]

# ---- 상태 전이 ----
def set_running(db: Session, output_id: str) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING",):
        raise ValueError(f"Invalid state transition: {obj.state} -> RUNNING")
    obj.state = "RUNNING"
    db.add(obj); db.commit(); db.refresh(obj)
    return _to_out(obj)

def set_failed(db: Session, output_id: str, reasons: Optional[List[str]] = None) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    obj.state = "FAILED"
    obj.kind = None
    obj.denied_reasons = reasons or []
    obj.completed_at = _now_iso()
    db.add(obj); db.commit(); db.refresh(obj)
    return _to_out(obj)

def set_complete_denied(db: Session, output_id: str, reasons: Optional[List[str]] = None, assumptions: Optional[List[str]] = None) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(DENIED)")
    obj.state = "COMPLETE"; obj.kind = "DENIED"
    obj.denied_reasons = reasons or []; obj.assumptions = assumptions or []
    obj.completed_at = _now_iso()
    db.add(obj); db.commit(); db.refresh(obj)
    return _to_out(obj)

def set_complete_answer(db: Session, output_id: str, answer_text: str, models_meta: Optional[dict] = None, assumptions: Optional[List[str]] = None) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(ANSWER)")
    obj.state = "COMPLETE"; obj.kind = "ANSWER"; obj.answer = answer_text
    obj.models = (obj.models or {}) | (models_meta or {})
    obj.assumptions = assumptions or []; obj.completed_at = _now_iso()
    db.add(obj); db.commit(); db.refresh(obj)
    return _to_out(obj)

def set_complete_processed(db: Session, output_id: str, impact: Optional[dict], prediction: Optional[dict], delta_stats: Optional[dict], models_meta: Optional[dict] = None, assumptions: Optional[List[str]] = None) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(PROCESSED)")
    obj.state = "COMPLETE"; obj.kind = "PROCESSED"
    obj.impact = impact or {}; obj.prediction = prediction or {}; obj.delta_stats = delta_stats or {}
    obj.models = (obj.models or {}) | (models_meta or {}); obj.assumptions = assumptions or []
    obj.completed_at = _now_iso()
    db.add(obj); db.commit(); db.refresh(obj)
    return _to_out(obj)

# ---- apply ----
def apply(db: Session, output_id: str) -> dict:
    out = db.get(OutputModel, output_id)
    if not out:
        raise ValueError("Output not found")
    if out.state != "COMPLETE" or out.kind != "PROCESSED":
        raise ValueError("Only COMPLETE & PROCESSED outputs can be applied")
    turn = db.get(TurnModel, out.turn_id)
    if not turn:
        raise ValueError("Turn not found")
    return {"applied": True}