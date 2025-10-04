from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timezone

from app.models import Turn as TurnModel, TurnState as TurnStateModel
from app.schemas import TurnCreate, TurnUpdateStats, TurnOut, TurnState, Stats

def now_iso():
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

ALLOWED_NEXT: Dict[TurnState, List[TurnState]] = {
    TurnState.DRAFT: [TurnState.BRIEFED, TurnState.COMMAND_PENDING],
    TurnState.BRIEFED: [TurnState.COMMAND_PENDING],
    TurnState.COMMAND_PENDING: [TurnState.VALIDATING, TurnState.REJECTED],
    TurnState.VALIDATING: [TurnState.VALIDATED, TurnState.REJECTED],
    TurnState.VALIDATED: [TurnState.COST_ESTIMATED],
    TurnState.COST_ESTIMATED: [TurnState.BUDGET_OK, TurnState.REJECTED],
    TurnState.BUDGET_OK: [TurnState.SIMULATED],
    TurnState.SIMULATED: [TurnState.APPLIED],
    TurnState.REJECTED: [TurnState.COMMAND_PENDING],
    TurnState.APPLIED: [],
}

def _new_id(prefix: str = "t") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _children_ids(obj: TurnModel) -> List[str]:
    return [c.id for c in obj.children_rel] if obj.children_rel else []

def to_out(obj: TurnModel) -> TurnOut:
    # children은 관계에서 수집
    return TurnOut(
        id=obj.id,
        parent_id=obj.parent_id,
        branch_id=obj.branch_id,
        month=obj.month,
        state=TurnState(obj.state.value),
        stats=obj.stats or {},
        children=_children_ids(obj),
        created_at=obj.created_at,
        updated_at=obj.updated_at,
    )

# app/services/turn_service.py

def _normalize_optional_id(v: str | None) -> str | None:
    # "", "   " -> None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

def _get_or_create_root(db: Session, branch_id: str) -> TurnModel:
    root_id = f"t_root_{branch_id}"
    root = db.get(TurnModel, root_id)
    if root:
        return root
    created = now_iso()
    root = TurnModel(
        id=root_id,
        parent_id=None,
        branch_id=branch_id,
        month="2000-01",  # 더미(YYYY-MM 패턴 만족)
        state=TurnStateModel.DRAFT,
        stats={"notes": ["root"]},
        created_at=created,
        updated_at=created,
    )
    db.add(root)
    db.commit()
    db.refresh(root)
    return root

def create_turn(db: Session, body: TurnCreate) -> TurnOut:
    tid = _new_id("t")
    created = now_iso()

    # 1) 입력 정규화
    parent_id = _normalize_optional_id(body.parent_id)
    branch_id = _normalize_optional_id(body.branch_id) or "b_main"

    # 2) 부모/브랜치 처리
    if parent_id:
        parent = db.get(TurnModel, parent_id)
        if not parent:
            raise ValueError(f"Parent turn not found: {parent_id}")
        # 부모가 있으면 부모 브랜치 상속(요청이 명시했으면 그 값 우선)
        branch_id = body.branch_id or parent.branch_id
    else:
        # parent_id가 없으면 해당 브랜치 루트를 보증하고 부모로 지정
        root = _get_or_create_root(db, branch_id)
        parent_id = root.id

    # 3) stats dict화
    stats_dict = body.stats.model_dump(by_alias=True) if getattr(body.stats, "model_dump", None) else (body.stats or {})

    # 4) INSERT
    turn = TurnModel(
        id=tid,
        parent_id=parent_id,
        branch_id=branch_id,
        month=body.month,
        state=TurnStateModel(body.state.value),
        stats=stats_dict,
        created_at=created,
        updated_at=created,
    )
    db.add(turn)
    try:
        db.flush()
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise ValueError("Failed to create Turn (constraint violation): " + str(e)) from e

    db.refresh(turn)

    # 5) 응답
    return TurnOut(
        id=turn.id,
        parent_id=turn.parent_id,
        branch_id=turn.branch_id,
        month=turn.month,
        state=TurnState(turn.state.value),
        stats=turn.stats or {},
        children=[c.id for c in turn.children_rel] if turn.children_rel else [],
        created_at=turn.created_at,
        updated_at=turn.updated_at,
    )

def get_turn(db: Session, turn_id: str) -> Optional[TurnOut]:
    obj = db.get(TurnModel, turn_id)
    return to_out(obj) if obj else None

def list_turns(
    db: Session,
    branch_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    month: Optional[str] = None,
    state: Optional[TurnState] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> List[TurnOut]:
    q = db.query(TurnModel)
    if branch_id:
        q = q.filter(TurnModel.branch_id == branch_id)
    if parent_id:
        q = q.filter(TurnModel.parent_id == parent_id)
    if month:
        q = q.filter(TurnModel.month == month)
    if state:
        q = q.filter(TurnModel.state == TurnStateModel(state.value))

    # 단순 커서: id 이후
    if cursor:
        q = q.filter(TurnModel.id > cursor)
    q = q.order_by(TurnModel.id).limit(limit)

    return [to_out(o) for o in q.all()]

def update_stats(db: Session, turn_id: str, body: TurnUpdateStats) -> Optional[TurnOut]:
    obj = db.get(TurnModel, turn_id)
    if not obj:
        return None
    obj.stats = body.stats.model_dump(by_alias=True)
    obj.updated_at = now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return to_out(obj)

def move_state(db: Session, turn_id: str, next_state: TurnState) -> Optional[TurnOut]:
    obj = db.get(TurnModel, turn_id)
    if not obj:
        return None
    cur = TurnState(obj.state.value)
    if next_state not in ALLOWED_NEXT.get(cur, []):
        # 서비스 레이어에서 None 대신 명시적 예외를 던져도 됨
        raise ValueError(f"Invalid state transition: {cur} -> {next_state}")
    obj.state = TurnStateModel(next_state.value)
    obj.updated_at = now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return to_out(obj)

def create_child_same_branch(db: Session, parent_id: str, body: TurnCreate) -> Optional[TurnOut]:
    parent = db.get(TurnModel, parent_id)
    if not parent:
        return None
    child_body = TurnCreate(
        parent_id=parent.id,
        branch_id=parent.branch_id,
        month=body.month,
        state=body.state,
        stats=body.stats,
    )
    return create_turn(db, child_body)

def create_branch_child(db: Session, parent_id: str, body: TurnCreate) -> Optional[TurnOut]:
    parent = db.get(TurnModel, parent_id)
    if not parent:
        return None
    new_branch_id = f"b_{uuid.uuid4().hex[:6]}"
    child_body = TurnCreate(
        parent_id=parent.id,
        branch_id=new_branch_id,
        month=body.month,
        state=body.state,
        stats=body.stats,
    )
    return create_turn(db, child_body)