# app/services/crop_stats_service.py
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import List
from app.models.session_crop_stats import SessionCropStats, Crop
from app.schemas.crop_stats import SessionCropStatsCreate, SessionCropStatsUpsert

def list_(db: Session, session_id: str) -> List[SessionCropStats]:
    """세션의 모든 작물 통계 조회"""
    return (
        db.query(SessionCropStats)
        .filter_by(session_id=session_id)
        .order_by(SessionCropStats.crop.asc())
        .all()
    )

def get_one(db: Session, session_id: str, crop: Crop) -> SessionCropStats | None:
    """세션의 특정 작물 통계 조회"""
    return (
        db.query(SessionCropStats)
        .filter_by(session_id=session_id, crop=crop)
        .first()
    )

def upsert_one(db: Session, session_id: str, body: SessionCropStatsUpsert) -> SessionCropStats:
    """세션의 작물 통계 생성 또는 업데이트"""
    row = get_one(db, session_id, body.crop)
    if row is None:
        # 생성
        row = SessionCropStats(session_id=session_id, crop=body.crop)
        db.add(row)

    # 업데이트 모드에 따라 처리
    if body.mode == "absolute":
        # 절대값 모드: 덮어쓰기
        row.cumulative_production_tonnes = body.cumulative_production_tonnes
        row.co2e_kg = body.co2e_kg
        row.water_m3 = body.water_m3
        row.fert_n_kg = body.fert_n_kg
        row.fert_p_kg = body.fert_p_kg
        row.fert_k_kg = body.fert_k_kg
    else:
        # delta 모드: 누적값 증가
        row.cumulative_production_tonnes += body.cumulative_production_tonnes
        row.co2e_kg += body.co2e_kg
        row.water_m3 += body.water_m3
        row.fert_n_kg += body.fert_n_kg
        row.fert_p_kg += body.fert_p_kg
        row.fert_k_kg += body.fert_k_kg

    row.last_event_at = datetime.now(tz=timezone.utc)
    db.commit()
    db.refresh(row)
    return row

def batch_upsert(db: Session, session_id: str, bodies: List[SessionCropStatsUpsert]) -> List[SessionCropStats]:
    """여러 작물 통계를 한 번에 업데이트"""
    out = []
    for b in bodies:
        out.append(upsert_one(db, session_id, b))
    return out

def delete_one(db: Session, session_id: str, crop: Crop) -> bool:
    """세션의 특정 작물 통계 삭제"""
    row = get_one(db, session_id, crop)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True

def delete_all(db: Session, session_id: str) -> int:
    """세션의 모든 작물 통계 삭제"""
    count = (
        db.query(SessionCropStats)
        .filter_by(session_id=session_id)
        .delete()
    )
    db.commit()
    return count
