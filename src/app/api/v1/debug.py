# app/api/v1/debug.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/db")
def debug_db(db: Session = Depends(get_db)):
    row = db.execute("""
        SELECT current_database() AS db,
               current_user AS usr,
               inet_server_addr() AS host,
               inet_server_port() AS port,
               current_schema() AS schema
    """).mappings().one()
    return dict(row)