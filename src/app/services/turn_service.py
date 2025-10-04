from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, union_all
from sqlalchemy.exc import IntegrityError, NoResultFound  # 추가
from typing import List, Optional, Dict, Any, Deque
from collections import deque
import uuid
from datetime import datetime, timezone

from app.models import Turn as TurnModel, TurnState as TurnStateModel
from app.schemas import TurnCreate, TurnUpdateStats, TurnOut, TurnState, Stats

# app/services/turn_service.py (상단 유틸 근처)
import uuid

def _new_session_id(prefix: str = "s") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

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

def _sibling_count(db: Session, parent_id: str) -> int:
    return db.query(TurnModel).filter(TurnModel.parent_id == parent_id).count()

def _new_branch_id() -> str:
    return f"b_{uuid.uuid4().hex[:6]}"

def to_out(obj: TurnModel) -> TurnOut:
    return TurnOut(
        id=obj.id,
        parent_id=obj.parent_id,
        branch_id=obj.branch_id,
        session_id=obj.session_id,         # ← 추가
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

# app/services/turn_service.py

def create_turn(db: Session, body: TurnCreate) -> TurnOut:
    tid = _new_id("t")
    created = now_iso()

    parent_id = _normalize_optional_id(body.parent_id)
    req_branch_id = _normalize_optional_id(body.branch_id)

    if parent_id:
        parent = db.get(TurnModel, parent_id)
        if not parent:
            raise ValueError(f"Parent turn not found: {parent_id}")
        branch_id = req_branch_id or parent.branch_id
        session_id = parent.session_id           # ★ 부모의 세션 상속
    else:
        branch_id = req_branch_id or "b_main"
        root = _get_or_create_root(db, branch_id)
        parent_id = root.id
        session_id = root.session_id             # ★ 루트의 세션 사용

    month_val = created[:7]
    state_val = TurnStateModel.DRAFT
    stats_dict = {}

    turn = TurnModel(
        id=tid,
        parent_id=parent_id,
        branch_id=branch_id,
        session_id=session_id,                   # ★ 필수
        month=month_val,
        state=state_val,
        stats=stats_dict,
        created_at=created,
        updated_at=created,
    )
    db.add(turn)
    db.flush()
    db.commit()
    db.refresh(turn)

    return TurnOut(
        id=turn.id,
        parent_id=turn.parent_id,
        branch_id=turn.branch_id,
        session_id=turn.session_id,             # ★ 응답 포함
        month=turn.month,
        state=TurnState(turn.state.value),
        stats=turn.stats or {},
        children=[c.id for c in turn.children_rel] if turn.children_rel else [],
        created_at=turn.created_at,
        updated_at=turn.updated_at,
    )
    
def _get_or_create_root(db: Session, branch_id: str) -> TurnModel:
    root_id = f"t_root_{branch_id}"
    root = db.get(TurnModel, root_id)
    if root:
        return root

    created = now_iso()

    # 1) 새 세션 생성
    from app.models import Session as SessionModel
    sid = _new_session_id()
    sess = SessionModel(id=sid, title=None)
    db.add(sess)
    db.flush()  # sid 확보

    # 2) 루트 생성 (session_id 연결)
    root = TurnModel(
        id=root_id,
        parent_id=None,
        branch_id=branch_id,
        session_id=sid,              # ★ 중요
        month="2000-01",
        state=TurnStateModel.DRAFT,
        stats={"notes": ["root"]},
        created_at=created,
        updated_at=created,
    )
    db.add(root)
    db.commit()
    db.refresh(root)
    return root

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
    # session_id는 create_turn에서 부모로부터 상속됨
    child_body = TurnCreate(
        parent_id=parent.id,
        branch_id=parent.branch_id,
    )
    return create_turn(db, child_body)

def create_branch_child(db: Session, parent_id: str, body: TurnCreate) -> Optional[TurnOut]:
    parent = db.get(TurnModel, parent_id)
    if not parent:
        return None
    new_branch_id = f"b_{uuid.uuid4().hex[:6]}"
    # session_id는 그대로 상속(브랜치만 새로) → create_turn에서 자동 상속
    child_body = TurnCreate(
        parent_id=parent.id,
        branch_id=new_branch_id,
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

def _serialize_turn(obj: TurnModel) -> dict:
    """트리 JSON 직렬화용 공통 포맷"""
    return {
        "id": obj.id,
        "parent_id": obj.parent_id,
        "branch_id": obj.branch_id,
        "session_id": obj.session_id,
        "month": obj.month,
        "state": getattr(obj.state, "value", obj.state),
        "stats": obj.stats or {},
        "created_at": obj.created_at,
        "updated_at": obj.updated_at,
        "children": [],   # 이후 채움
    }

def get_turn_tree(
    db: Session,
    root_turn_id: str,
    *,
    session_id: str,                 # ★ 추가: 반드시 세션 지정
    max_nodes: int = 5000,
    use_recursive_cte: bool = False
) -> dict:
    """
    root_turn_id를 루트로 하는 서브트리를 JSON으로 반환.
    - 순환참조 가드
    - 노드 상한 (max_nodes)
    - use_recursive_cte=True 이면 Postgres에서 재귀 CTE로 일괄 로딩(고성능)
    """
    root = db.get(TurnModel, root_turn_id)
    if not root:
        raise ValueError(f"Turn not found: {root_turn_id}")

    if root.session_id != session_id:
        raise ValueError(
            f"Turn {root_turn_id} does not belong to session {session_id}"
        )
    if use_recursive_cte:
        # ---------- 고급: Postgres 재귀 CTE ----------
        # 모든 하위 노드를 한 번에 가져온 뒤, 파이썬에서 트리로 조립
        T = TurnModel.__table__
        base = select(
            T.c.id, T.c.parent_id, T.c.branch_id, T.c.month, T.c.state,
            T.c.stats, T.c.created_at, T.c.updated_at, T.c.session_id  # ← 선택: 디버깅/확인용
        ).where(T.c.id == root_turn_id)

        base_cte = base.cte(name="turn_subtree", recursive=True)

        # ★ 자식 확장 시 같은 session만 허용
        rec = select(
            T.c.id, T.c.parent_id, T.c.branch_id, T.c.month, T.c.state,
            T.c.stats, T.c.created_at, T.c.updated_at, T.c.session_id
        ).where(
            (T.c.parent_id == base_cte.c.id) &
            (T.c.session_id == root_sid)       # ★ 핵심 조건
        )

        subtree_cte = base_cte.union_all(rec)

        rows = db.execute(select(subtree_cte)).fetchall()
        if len(rows) > max_nodes:
            raise ValueError(f"Subtree too large (> {max_nodes} nodes)")

        # id -> 노드(dict) 매핑 생성
        nodes: Dict[str, dict] = {}
        children_map: Dict[str, list] = {}
        for r in rows:
            # r는 Row(tuple) 형태: (id, parent_id, branch_id, month, state, stats, created_at, updated_at)
            node = {
                "id": r.id,
                "parent_id": r.parent_id,
                "branch_id": r.branch_id,
                "session_id": r.session_id,
                "month": r.month,
                "state": getattr(r.state, "value", r.state) if hasattr(r, "state") else r.state,
                "stats": r.stats or {},
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "children": [],
            }
            nodes[r.id] = node
            if r.parent_id:
                children_map.setdefault(r.parent_id, []).append(r.id)

        # children 연결
        for pid, kids in children_map.items():
            parent_node = nodes.get(pid)
            if parent_node:
                for kid_id in kids:
                    child_node = nodes[kid_id]
                    parent_node["children"].append(child_node)

        return nodes[root_turn_id]

    else:
        # ---------- 기본: ORM로 BFS (joinedload로 children 미리 로딩 시 N+1 완화) ----------
        # children_rel 관계명이 프로젝트마다 다를 수 있으니, 현재 코드 기준 children_rel 사용
        visited: set[str] = set()
        order: list[TurnModel] = []
        q: Deque[str] = deque([root_turn_id])

        while q:
            cur_id = q.popleft()
            if cur_id in visited:
                # 순환 참조 방지
                continue
            visited.add(cur_id)

            cur = (
                db.query(TurnModel)
                  .options(joinedload(TurnModel.children_rel))
                  .filter(TurnModel.id == cur_id)
                  .one_or_none()
            )
            if not cur:
                # 누락된 자식 id가 DB에 없을 수 있으므로 스킵
                continue

            order.append(cur)
            if len(visited) > max_nodes:
                raise ValueError(f"Subtree too large (> {max_nodes} nodes)")

            # 큐에 자식 push
            for ch in (cur.children_rel or []):
                if ch.session_id == root_sid and ch.id not in visited:  # ★ 같은 session만
                    q.append(ch.id)

        # id -> dict 노드
        nodes: Dict[str, dict] = {obj.id: _serialize_turn(obj) for obj in order}

        # children 연결
        for obj in order:
            node = nodes[obj.id]
            for ch in (obj.children_rel or []):
                node["children"].append(nodes[ch.id])

        return nodes[root_turn_id]