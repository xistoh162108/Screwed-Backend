# app/services/command_service.py
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timezone

from app.models import Command as CommandModel, Turn as TurnModel, Output as OutputModel
from app.schemas.command import (
    CommandCreate, CommandCreateOut, CommandOut, CommandValidateIn, CommandCostIn
)
from app.services import output_service
from app.services.command_task import validate_command_task  # Celery task 등록함수

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _to_out(obj: CommandModel) -> CommandOut:
    # created_at을 ISO Z로 보장
    created_at = obj.created_at
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        created_at = created_at.isoformat().replace("+00:00", "Z")
    else:
        created_at = str(created_at)

    return CommandOut(
        id=obj.id,
        turn_id=obj.turn_id,
        text=obj.text,
        validity=obj.validity,
        cost=obj.cost,
        created_at=created_at,
    )

def create_and_enqueue_validation(db: Session, turn_id: str, body: CommandCreate) -> CommandCreateOut:
    # 1) Turn 존재 확인
    turn = db.get(TurnModel, turn_id)
    if not turn:
        raise ValueError("Turn not found")

    # 2) Command 생성
    cid = _new_id("c")
    cmd = CommandModel(
        id=cid,
        turn_id=turn_id,
        text=body.text,
        created_at=_now_iso(),   # 문자열로 저장해도 무방
        # 필요 시 payload 저장 칼럼이 있으면 여기에 body.payload 반영
        payload=body.payload or {},
    )
    db.add(cmd)
    db.commit()

    db.refresh(cmd)
    # 3) Output(PENDING) 생성 (이 Command의 검증 결과를 담는 그릇)
    out_id_obj = output_service.create_from_command(db, cid)  # 멱등 처리 포함
    output_id = out_id_obj.output_id

    # 4) Celery 큐에 태스크 enqueue
    async_result = validate_command_task.delay(command_id=cid, output_id=output_id)

    # 5) task_id 저장(선택) — Output.models에 task_id 넣기
    from app.models import Output as OutputModel
    out = db.get(OutputModel, output_id)
    if out:
        models = out.models or {}
        models["task_id"] = async_result.id
        out.models = models
        db.add(out)
        db.commit()

    # 6) 최종 반환
    return CommandCreateOut(
        command_id=cid,
        output_id=output_id,
        task_id=async_result.id,
    )

# ====== 이하 기존 동작 유지 (조회/리스트/수동세팅) ======
def get(db: Session, cmd_id: str) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    return _to_out(obj) if obj else None

def list_by_turn(db: Session, turn_id: str, limit: int = 100, cursor: Optional[str] = None) -> List[CommandOut]:
    q = db.query(CommandModel).filter(CommandModel.turn_id == turn_id).order_by(CommandModel.id)
    if cursor:
        q = q.filter(CommandModel.id > cursor)
    rows = q.limit(limit).all()
    return [_to_out(o) for o in rows]

def set_validity(db: Session, cmd_id: str, v: CommandValidateIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.validity = {"is_valid": v.is_valid, "score": v.score, "reasons": v.reasons or []}
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)

def set_cost(db: Session, cmd_id: str, c: CommandCostIn) -> Optional[CommandOut]:
    obj = db.get(CommandModel, cmd_id)
    if not obj:
        return None
    obj.cost = {"estimate": c.estimate, "currency": c.currency, "breakdown": c.breakdown or []}
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _to_out(obj)