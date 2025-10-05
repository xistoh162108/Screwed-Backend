# app/api/v1/session_crops.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from sqlalchemy.orm import Session as DBSession
from app.db.session import get_db
from app.schemas.crop_stats import SessionCropStatsOut, SessionCropStatsUpsert
from app.models.session_crop_stats import Crop
from app.services import crop_stats_service, session_service

router = APIRouter(prefix="/sessions", tags=["sessions-crops"])

@router.get("/{session_id}/crops", response_model=List[SessionCropStatsOut])
def list_crops(session_id: str, db: DBSession = Depends(get_db)):
    """세션의 모든 작물 통계 조회"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    return crop_stats_service.list_(db, session_id)

@router.get("/{session_id}/crops/{crop}", response_model=SessionCropStatsOut)
def get_crop_stats(session_id: str, crop: Crop, db: DBSession = Depends(get_db)):
    """세션의 특정 작물 통계 조회"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    row = crop_stats_service.get_one(db, session_id, crop)
    if not row:
        raise HTTPException(404, f"Crop stats for {crop} not found")
    return row

@router.put("/{session_id}/crops/{crop}", response_model=SessionCropStatsOut)
def put_crop_stats(
    session_id: str,
    crop: Crop,
    body: SessionCropStatsUpsert,
    db: DBSession = Depends(get_db)
):
    """세션의 작물 통계 생성 또는 전체 업데이트 (upsert)"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    # path의 crop을 강제 적용 (신뢰성)
    body.crop = crop
    return crop_stats_service.upsert_one(db, session_id, body)

@router.patch("/{session_id}/crops/{crop}", response_model=SessionCropStatsOut)
def patch_crop_stats(
    session_id: str,
    crop: Crop,
    body: SessionCropStatsUpsert,
    db: DBSession = Depends(get_db)
):
    """세션의 작물 통계 부분 업데이트"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    # path의 crop을 강제 적용
    body.crop = crop
    return crop_stats_service.upsert_one(db, session_id, body)

@router.put("/{session_id}/crops:batch", response_model=List[SessionCropStatsOut])
def put_crops_batch(
    session_id: str,
    bodies: List[SessionCropStatsUpsert],
    db: DBSession = Depends(get_db)
):
    """여러 작물 통계를 한 번에 업데이트 (batch upsert)"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    return crop_stats_service.batch_upsert(db, session_id, bodies)

@router.delete("/{session_id}/crops/{crop}", status_code=status.HTTP_204_NO_CONTENT)
def delete_crop_stats(session_id: str, crop: Crop, db: DBSession = Depends(get_db)):
    """세션의 특정 작물 통계 삭제"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    ok = crop_stats_service.delete_one(db, session_id, crop)
    if not ok:
        raise HTTPException(404, f"Crop stats for {crop} not found")

@router.delete("/{session_id}/crops", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_crop_stats(session_id: str, db: DBSession = Depends(get_db)):
    """세션의 모든 작물 통계 삭제"""
    # 세션 존재 확인
    if not session_service.get(db, session_id):
        raise HTTPException(404, "Session not found")

    crop_stats_service.delete_all(db, session_id)
