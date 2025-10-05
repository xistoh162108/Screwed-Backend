from fastapi import APIRouter, Depends, HTTPException, Query, status, Header
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db, engine
from app.db.base import Base
from app.schemas import TurnCreate, TurnOut, TurnUpdateStats, TurnState
from app.services import turn_service
from app.schemas.turns_crash import CrashNodeInput, ConflictOut  # ★ 추가

router = APIRouter(prefix="/turns", tags=["turns"])

# 최초 테이블 생성(간단 데모용: 운영은 Alembic 권장)
Base.metadata.create_all(bind=engine)

@router.post("", response_model=TurnOut, status_code=status.HTTP_201_CREATED)
def create_turn(body: TurnCreate, db: Session = Depends(get_db)):
    return turn_service.create_turn(db, body)

@router.get("/{turn_id}", response_model=TurnOut)
def get_turn(turn_id: str, db: Session = Depends(get_db)):
    t = turn_service.get_turn(db, turn_id)
    if not t:
        raise HTTPException(404, "Turn not found")
    return t

@router.get("", response_model=List[TurnOut])
def list_turns(
    branch_id: Optional[str] = Query(None),
    parent_id: Optional[str] = Query(None),
    month: Optional[str] = Query(None),
    state: Optional[TurnState] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    return turn_service.list_turns(db, branch_id, parent_id, month, state, limit, cursor)

@router.patch("/{turn_id}/stats", response_model=TurnOut)
def update_stats(turn_id: str, body: TurnUpdateStats, db: Session = Depends(get_db)):
    t = turn_service.update_stats(db, turn_id, body)
    if not t:
        raise HTTPException(404, "Turn not found")
    return t

@router.post("/{turn_id}/state/{next_state}", response_model=TurnOut)
def move_state(turn_id: str, next_state: TurnState, db: Session = Depends(get_db)):
    try:
        t = turn_service.move_state(db, turn_id, next_state)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not t:
        raise HTTPException(404, "Turn not found")
    return t

@router.post("/{turn_id}/children", response_model=TurnOut, status_code=201)
def create_child(turn_id: str, body: TurnCreate, db: Session = Depends(get_db)):
    t = turn_service.create_child_same_branch(db, turn_id, body)
    if not t:
        raise HTTPException(404, "Parent not found")
    return t

@router.post("/{turn_id}/branch", response_model=TurnOut, status_code=201)
def create_branch(turn_id: str, body: TurnCreate, db: Session = Depends(get_db)):
    t = turn_service.create_branch_child(db, turn_id, body)
    if not t:
        raise HTTPException(404, "Parent not found")
    return t

@router.get("/{turn_id}/tree")
def get_turn_tree_endpoint(
    turn_id: str,
    session_id: str = Query(..., description="Session ID to constrain the tree"),  # ★ 필수
    max_nodes: int = Query(5000, ge=1, le=20000),
    use_recursive_cte: bool = Query(False),
    db: Session = Depends(get_db),
):
    try:
        return turn_service.get_turn_tree(
            db, turn_id,
            session_id=session_id,               # ★ 전달
            max_nodes=max_nodes,
            use_recursive_cte=use_recursive_cte,
        )
    except ValueError as e:
        msg = str(e).lower()
        # 루트 미존재/세션 불일치 → 404가 UX상 더 직관적
        if "not found" in msg or "does not belong" in msg:
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
    
@router.post("/crash",
             responses={
                 200: {"model": TurnOut},
                 201: {"model": TurnOut},
                 409: {"model": ConflictOut},
                 412: {"description": "ETag mismatch"},
             })
def crash_turn(
    body: CrashNodeInput,
    db: Session = Depends(get_db),
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
):
    """
    current_id가 최신이면 같은 브랜치로 자식 턴 생성.
    최신이 아니면 409 반환. body.resolution == 'FORK' 이면 포크 생성 허용.
    If-Match 헤더가 있으면 ETag 불일치 시 412.
    """
    try:
        created, http_status = turn_service.crash_turn(db, body, if_match)
        # FastAPI에서 상태코드 동적 반환
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status, content=created.model_dump())
    except turn_service.ETagMismatch:
        raise HTTPException(status_code=412, detail="ETag mismatch")
    except turn_service.TimelineConflict as e:
        # 충돌 세부정보 제공
        raise HTTPException(
            status_code=409,
            detail={
                "error": "TIMELINE_CONFLICT",
                "message": str(e),
                "latest_id": e.latest_id,
                "latest_etag": e.latest_etag,
                "resolution": "FORK",
                "branch_id": e.branch_id,
            },
        )
    except ValueError as e:
        # 예: current_id 없음 등
        raise HTTPException(status_code=404, detail=str(e))