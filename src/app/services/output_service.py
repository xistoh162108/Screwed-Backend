from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
from datetime import datetime, timezone

from app.models import (
    Output as OutputModel,
    Command as CommandModel,
    Turn as TurnModel,
)
from app.schemas.output import (
    OutputCreateIn,
    OutputIdOut,
    OutputOut,
)

# ---------- 공통 유틸 ----------
from datetime import datetime, timezone

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _normalize_iso(dt_val):
    """
    DB에 저장된 값이 str 이든 datetime 이든 안전하게 'YYYY-MM-DDTHH:MM:SSZ'로 반환
    """
    if dt_val is None:
        return None
    if isinstance(dt_val, str):
        # 이미 문자열이면 +00:00 -> Z 정규화만 수행
        return dt_val.replace("+00:00", "Z")
    # datetime이면 Z로 직렬화
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
        created_at=_normalize_iso(obj.created_at),                 # ← 변경
        completed_at=_normalize_iso(obj.completed_at),            # ← 변경
    )

# ---------- 생성 ----------
def create_from_command(
    db: Session,
    command_id: str,
    body: Optional[OutputCreateIn] = None,
) -> OutputIdOut:
    cmd = db.get(CommandModel, command_id)
    if not cmd:
        raise ValueError("Command not found")

    # Command별 Output은 1개 정책이면, 중복 보호(UNIQUE 제약 권장)
    existing = db.query(OutputModel).filter(OutputModel.command_id == command_id).first()
    if existing:
        # 멱등 처리: 기존 id 반환
        return OutputIdOut(output_id=existing.id)

    oid = _new_id("o")
    models = None
    if body:
        models = {
            "fine_tuned_model_id": body.fine_tuned_model_id,
            "foundation_model_id": body.foundation_model_id,
            "options": body.options or {},
        }

    # 생성 시
    out = OutputModel(
        id=oid,
        command_id=command_id,
        turn_id=cmd.turn_id,
        state="PENDING",
        kind=None,
        answer=None,
        impact=None,
        prediction=None,
        delta_stats=None,
        models=models,
        assumptions=None,
        denied_reasons=None,
        created_at=_now_iso(),        # ← datetime 말고 str 저장
        completed_at=None,
    )
    db.add(out)
    db.commit()
    return OutputIdOut(output_id=oid)

# ---------- 조회 ----------
def get(db: Session, output_id: str) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    return _to_out(obj) if obj else None

def list_by_turn(
    db: Session, turn_id: str, limit: int = 100, cursor: Optional[str] = None
) -> List[OutputOut]:
    q = (
        db.query(OutputModel)
        .filter(OutputModel.turn_id == turn_id)
        .order_by(OutputModel.id)
    )
    if cursor:
        q = q.filter(OutputModel.id > cursor)
    rows = q.limit(limit).all()
    return [_to_out(o) for o in rows]

# ---------- 상태 전이 헬퍼 ----------
def set_running(db: Session, output_id: str) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING",):  # 간단한 가드
        raise ValueError(f"Invalid state transition: {obj.state} -> RUNNING")
    obj.state = "RUNNING"
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

def set_failed(db: Session, output_id: str, reasons: Optional[List[str]] = None) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    obj.state = "FAILED"
    obj.kind = None
    obj.denied_reasons = reasons or []
    obj.completed_at = _now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

# ---------- COMPLETE 전이(3가지 케이스) ----------
def set_complete_denied(
    db: Session,
    output_id: str,
    reasons: Optional[List[str]] = None,
    assumptions: Optional[List[str]] = None,
) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(DENIED)")
    obj.state = "COMPLETE"
    obj.kind = "DENIED"
    obj.denied_reasons = reasons or []
    obj.assumptions = assumptions or []
    obj.completed_at = _now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

def set_complete_answer(
    db: Session,
    output_id: str,
    answer_text: str,
    models_meta: Optional[dict] = None,
    assumptions: Optional[List[str]] = None,
) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(ANSWER)")
    obj.state = "COMPLETE"
    obj.kind = "ANSWER"
    obj.answer = answer_text
    obj.models = (obj.models or {}) | (models_meta or {})
    obj.assumptions = assumptions or []
    obj.completed_at = _now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

def set_complete_processed(
    db: Session,
    output_id: str,
    impact: Optional[dict],
    prediction: Optional[dict],
    delta_stats: Optional[dict],
    models_meta: Optional[dict] = None,
    assumptions: Optional[List[str]] = None,
) -> Optional[OutputOut]:
    obj = db.get(OutputModel, output_id)
    if not obj:
        return None
    if obj.state not in ("PENDING", "RUNNING"):
        raise ValueError(f"Invalid state transition: {obj.state} -> COMPLETE(PROCESSED)")
    obj.state = "COMPLETE"
    obj.kind = "PROCESSED"
    obj.impact = impact or {}
    obj.prediction = prediction or {}
    obj.delta_stats = delta_stats or {}
    obj.models = (obj.models or {}) | (models_meta or {})
    obj.assumptions = assumptions or []
    obj.completed_at = _now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

# ---------- apply(커밋) ----------
def apply(db: Session, output_id: str) -> dict:
    """
    PROCESSED & COMPLETE 상태만 커밋 허용.
    - 여기서 실제 머니 반영/스탯 갱신/다음 턴 생성 등을 처리.
    - 턴/원장/브랜치 정책은 프로젝트별로 다르므로 TODO 훅 제공.
    """
    out = db.get(OutputModel, output_id)
    if not out:
        raise ValueError("Output not found")
    if out.state != "COMPLETE" or out.kind != "PROCESSED":
        raise ValueError("Only COMPLETE & PROCESSED outputs can be applied")

    turn = db.get(TurnModel, out.turn_id)
    if not turn:
        raise ValueError("Turn not found")

    # ---- TODO: 아래 3가지는 서비스 정책에 맞게 구현하세요 ----
    # 1) 원장/머니 업데이트 (turn.stats.money, ledger 기록 등)
    # 2) turn.stats에 prediction/delta_stats 반영
    # 3) 자식 턴 자동 생성(다음 달) 또는 현재 턴을 APPLIED로 마킹
    #
    # 예시(개략):
    # turn.stats = _apply_stats(turn.stats, out.delta_stats, out.prediction)
    # child_id = _create_next_turn(db, parent_turn=turn, prediction=out.prediction)
    # db.add(turn); db.commit()

    result = {
        "applied": True,
        # "child_turn_id": child_id,
        # "snapshot_before": ...,
        # "snapshot_after":  ...,
    }
    return result