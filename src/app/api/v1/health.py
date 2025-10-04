# app/api/v1/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db

router = APIRouter(tags=["health"])

@router.get("/healthz", summary="(DB & Server) Health Check")
def healthz(db: Session = Depends(get_db)):
    db.execute("SELECT 1")
    return {"status": "ok"}