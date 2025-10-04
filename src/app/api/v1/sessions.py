from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session as DBSession
from typing import List, Optional

from app.db.session import get_db, engine
from app.db.base import Base
from app.schemas import SessionCreate, SessionUpdate, SessionOut
from app.services import session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])

# 데모용 자동 테이블 생성 (운영은 Alembic 권장)
Base.metadata.create_all(bind=engine)

@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(body: SessionCreate, db: DBSession = Depends(get_db)):
    try:
        return session_service.create(db, body)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    s = session_service.get(db, session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s

@router.get("", response_model=List[SessionOut])
def list_sessions(limit: int = Query(50, ge=1, le=200), cursor: Optional[str] = Query(None), db: DBSession = Depends(get_db)):
    return session_service.list_(db, limit, cursor)

@router.patch("/{session_id}", response_model=SessionOut)
def update_session(session_id: str, body: SessionUpdate, db: DBSession = Depends(get_db)):
    s = session_service.update(db, session_id, body)
    if not s:
        raise HTTPException(404, "Session not found")
    return s

@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, db: DBSession = Depends(get_db)):
    ok = session_service.delete(db, session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return