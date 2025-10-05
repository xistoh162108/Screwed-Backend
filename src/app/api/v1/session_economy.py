# app/api/v1/session_economy.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from app.db.session import get_db
from app.schemas.economy import SessionEconomyCreate, SessionEconomyUpdate, SessionEconomyOut
from app.services import economy_service, session_service

router = APIRouter(prefix="/sessions", tags=["sessions-economy"])

@router.get("/{session_id}/economy", response_model=SessionEconomyOut)
def get_economy(session_id: str, db: DBSession = Depends(get_db)):
    """세션의 경제 정보 조회"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    row = economy_service.get(db, session_id)
    if not row:
        raise HTTPException(404, "Economy not found for this session")
    return row

@router.put("/{session_id}/economy", response_model=SessionEconomyOut, status_code=status.HTTP_200_OK)
def put_economy(session_id: str, body: SessionEconomyCreate, db: DBSession = Depends(get_db)):
    """세션의 경제 정보 생성 또는 전체 업데이트 (upsert)"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    return economy_service.upsert(db, session_id, body)

@router.patch("/{session_id}/economy", response_model=SessionEconomyOut)
def patch_economy(session_id: str, body: SessionEconomyUpdate, db: DBSession = Depends(get_db)):
    """세션의 경제 정보 부분 업데이트"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    return economy_service.upsert(db, session_id, body)

@router.delete("/{session_id}/economy", status_code=status.HTTP_204_NO_CONTENT)
def delete_economy(session_id: str, db: DBSession = Depends(get_db)):
    """세션의 경제 정보 삭제"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    ok = economy_service.delete(db, session_id)
    if not ok:
        raise HTTPException(404, "Economy not found for this session")
