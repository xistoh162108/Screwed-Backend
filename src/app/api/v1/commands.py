# app/api/v1/commands.py
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db, engine
from app.db.base import Base
from app.schemas import CommandCreate, CommandOut, CommandValidateIn, CommandCostIn
from app.services import command_service

router = APIRouter(prefix="/turns/{turn_id}/commands", tags=["commands"])

# 테이블 생성(개발 편의; 운영은 Alembic 권장)
Base.metadata.create_all(bind=engine)

@router.post("", response_model=CommandOut, status_code=status.HTTP_201_CREATED)
def create_command(turn_id: str, body: CommandCreate, db: Session = Depends(get_db)):
    try:
        return command_service.create(db, turn_id, body)
    except ValueError as e:
        raise HTTPException(404, detail=str(e))

@router.get("", response_model=List[CommandOut])
def list_commands(turn_id: str, limit: int = Query(100, ge=1, le=500), cursor: Optional[str] = None, db: Session = Depends(get_db)):
    return command_service.list_by_turn(db, turn_id, limit, cursor)

@router.get("/{cmd_id}", response_model=CommandOut)
def get_command(turn_id: str, cmd_id: str, db: Session = Depends(get_db)):
    c = command_service.get(db, cmd_id)
    if not c or c.turn_id != turn_id:
        raise HTTPException(404, "Command not found")
    return c

@router.post("/{cmd_id}/validate", response_model=CommandOut)
def validate_command(turn_id: str, cmd_id: str, body: CommandValidateIn, db: Session = Depends(get_db)):
    c = command_service.set_validity(db, cmd_id, body)
    if not c or c.turn_id != turn_id:
        raise HTTPException(404, "Command not found")
    return c

@router.post("/{cmd_id}/estimate-cost", response_model=CommandOut)
def estimate_cost(turn_id: str, cmd_id: str, body: CommandCostIn, db: Session = Depends(get_db)):
    c = command_service.set_cost(db, cmd_id, body)
    if not c or c.turn_id != turn_id:
        raise HTTPException(404, "Command not found")
    return c