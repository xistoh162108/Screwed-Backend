# app/services/economy_service.py
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.models.session_economy import SessionEconomy
from app.schemas.economy import SessionEconomyCreate, SessionEconomyUpdate

def get(db: Session, session_id: str) -> SessionEconomy | None:
    """세션의 경제 정보 조회"""
    return db.query(SessionEconomy).filter_by(session_id=session_id).first()

def upsert(db: Session, session_id: str, body: SessionEconomyCreate | SessionEconomyUpdate) -> SessionEconomy:
    """세션 경제 정보 생성 또는 업데이트 (upsert)"""
    row = get(db, session_id)
    if row is None:
        # 생성
        row = SessionEconomy(session_id=session_id)
        db.add(row)

    # 업데이트
    if getattr(body, "currency", None) is not None:
        row.currency = body.currency
    if getattr(body, "balance", None) is not None:
        row.balance = body.balance

    row.updated_at = datetime.now(tz=timezone.utc)
    db.commit()
    db.refresh(row)
    return row

def delete(db: Session, session_id: str) -> bool:
    """세션 경제 정보 삭제"""
    row = get(db, session_id)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True
