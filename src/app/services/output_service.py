# app/services/output_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, NamedTuple

from sqlalchemy.orm import Session

from app.models import Output as OutputModel, Command as CommandModel
# Output 모델이 now_iso를 사용하므로 동일 형식으로 맞춥니다.
from datetime import datetime, timezone

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# --- 상태/종류 상수 (모델 주석과 일치) ---
STATE_PENDING = "PENDING"
STATE_RUNNING = "RUNNING"
STATE_COMPLETE = "COMPLETE"
STATE_FAILED  = "FAILED"

KIND_ANSWER   = "ANSWER"
KIND_DENIED   = "DENIED"
KIND_PROCESSED = "PROCESSED"

class CreateFromCommandResult(NamedTuple):
    output_id: str

# -------------------------------------------------------
# 1) Command로부터 Output 생성 (멱등). 없으면 만들고, 있으면 그대로 반환
#    Output 스키마상 turn_id/state 필수이므로 Command에서 turn_id를 가져와 채웁니다.
# -------------------------------------------------------
def create_from_command(db: Session, command_id: str) -> CreateFromCommandResult:
    # 이미 존재하면 재사용(멱등)
    existing = (
        db.query(OutputModel)
          .filter(OutputModel.command_id == command_id)
          .first()
    )
    if existing:
        return CreateFromCommandResult(output_id=existing.id)

    # Command 존재/turn_id 확보
    cmd = db.get(CommandModel, command_id)
    if not cmd:
        raise ValueError(f"Command not found: {command_id}")
    if not getattr(cmd, "turn_id", None):
        raise ValueError(f"Command.turn_id missing (command_id={command_id})")

    # Output id는 모델에 default가 없다면 여기서 생성 규칙을 맞춰주세요.
    # 예: o_<8hex>
    import uuid
    oid = f"o_{uuid.uuid4().hex[:8]}"

    out = OutputModel(
        id=oid,
        command_id=command_id,
        turn_id=cmd.turn_id,
        state=STATE_PENDING,
        kind=None,
        answer=None,
        impact=None,
        prediction=None,
        delta_stats=None,
        models={},
        assumptions=[],
        denied_reasons=None,
        created_at=_now_iso(),
        completed_at=None,
    )
    db.add(out)
    db.commit()
    db.refresh(out)
    return CreateFromCommandResult(output_id=out.id)

# -------------------------------------------------------
# 2) RUNNING 전이
# -------------------------------------------------------
def set_running(db: Session, output_id: str) -> None:
    out = db.get(OutputModel, output_id)
    if not out:
        raise ValueError(f"Output not found: {output_id}")
    out.state = STATE_RUNNING
    db.add(out)
    db.commit()

# -------------------------------------------------------
# 3) 정상 완료: 답변(ANSWER)
#    - state=COMPLETE, kind=ANSWER, answer/모델메타/가정/완료시각 설정
# -------------------------------------------------------
def set_complete_answer(
    db: Session,
    *,
    output_id: str,
    answer_text: str,
    models_meta: Optional[Dict[str, Any]] = None,
    assumptions: Optional[List[str]] = None,
    impact: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    delta_stats: Optional[Dict[str, Any]] = None,
) -> None:
    out = db.get(OutputModel, output_id)
    if not out:
        raise ValueError(f"Output not found: {output_id}")

    out.state = STATE_COMPLETE
    out.kind = KIND_ANSWER
    out.answer = answer_text
    out.models = models_meta or {}
    out.assumptions = assumptions or []
    out.denied_reasons = None
    # 선택 페이로드
    out.impact = impact
    out.prediction = prediction
    out.delta_stats = delta_stats

    out.completed_at = _now_iso()
    db.add(out)
    db.commit()

# -------------------------------------------------------
# 4) 정상 완료: 거절(DENIED)
#    - state=COMPLETE, kind=DENIED, 사유 기록
# -------------------------------------------------------
def set_complete_denied(
    db: Session,
    *,
    output_id: str,
    reasons: List[str],
    models_meta: Optional[Dict[str, Any]] = None,
    assumptions: Optional[List[str]] = None,
) -> None:
    out = db.get(OutputModel, output_id)
    if not out:
        raise ValueError(f"Output not found: {output_id}")

    out.state = STATE_COMPLETE
    out.kind = KIND_DENIED
    out.answer = None
    out.denied_reasons = reasons or ["denied"]
    out.models = models_meta or {}
    out.assumptions = assumptions or []
    # 선택 페이로드 초기화
    out.impact = None
    out.prediction = None
    out.delta_stats = None

    out.completed_at = _now_iso()
    db.add(out)
    db.commit()

# -------------------------------------------------------
# 5) 실패(예외/시스템 오류)
#    - state=FAILED, kind(None 유지), 사유를 denied_reasons에 담아두면 API에서 일관되게 확인 가능
# -------------------------------------------------------
def set_failed(
    db: Session,
    output_id: str,
    reasons: Optional[List[str]] = None,
    models_meta: Optional[Dict[str, Any]] = None,
) -> None:
    out = db.get(OutputModel, output_id)
    if not out:
        # 기록 불가 시 조용히 반환할지, 예외를 던질지는 팀 규칙에 맞추세요.
        return

    out.state = STATE_FAILED
    # 실패는 비즈니스 거절이 아니므로 kind는 유지(None 권장)
    out.answer = None
    out.denied_reasons = reasons or ["unexpected error"]
    if models_meta:
        out.models = (out.models or {}) | models_meta

    out.completed_at = _now_iso()
    db.add(out)
    db.commit()
    
# app/services/output_service.py (enqueue_output_from_command 내부만 수정)
def enqueue_output_from_command(db: Session, command_id: str) -> str:
    res = create_from_command(db, command_id)
    output_id = res.output_id

    # 지연 임포트로 순환 참조 방지
    from app.workers.validate import validate_command_task

    async_result = validate_command_task.delay(command_id=command_id, output_id=output_id)

    out = db.get(OutputModel, output_id)
    if out:
        models = out.models or {}
        models["task_id"] = async_result.id
        out.models = models
        db.add(out)
        db.commit()

    return output_id