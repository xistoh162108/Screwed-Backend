from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
# app/api/v1/commands.py
from app.schemas.command import (    
    CommandCreate,   
    CommandOut,
    CommandValidateIn,
    CommandCostIn,
    CommandCreateOut,                     
)

from app.services.command_service import (  # ← 서비스 함수도 명시 import
    create_and_enqueue_validation,       
    get as get_command_svc,
    set_validity,
    set_cost,
)
router = APIRouter(prefix="/turns/{turn_id}/commands", tags=["commands"])

@router.post("", response_model=CommandCreateOut, status_code=status.HTTP_201_CREATED)
def create_command(turn_id: str, body: CommandCreate, db: Session = Depends(get_db)):
    try:
        return create_and_enqueue_validation(db, turn_id, body)
    except ValueError as e:
        # 예: Turn not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # 기타 예외는 400/500 중 정책에 맞게
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[CommandOut])
def list_commands(
    turn_id: str,
    limit: int = Query(100, ge=1, le=500),
    cursor: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return command_service.list_by_turn(db, turn_id, limit, cursor)

@router.get("/{cmd_id}", response_model=CommandOut)
def get_command(turn_id: str, cmd_id: str, db: Session = Depends(get_db)):
    c = command_service.get(db, cmd_id)
    if not c or c.turn_id != turn_id:
        raise HTTPException(status_code=404, detail="Command not found")
    return c

@router.post("/{cmd_id}/validate", response_model=CommandOut)
def validate_command(turn_id: str, cmd_id: str, body: CommandValidateIn, db: Session = Depends(get_db)):
    c = command_service.set_validity(db, cmd_id, body)
    if not c or c.turn_id != turn_id:
        raise HTTPException(status_code=404, detail="Command not found")
    return c

@router.post("/{cmd_id}/estimate-cost", response_model=CommandOut)
def estimate_cost(turn_id: str, cmd_id: str, body: CommandCostIn, db: Session = Depends(get_db)):
    c = command_service.set_cost(db, cmd_id, body)
    if not c or c.turn_id != turn_id:
        raise HTTPException(status_code=404, detail="Command not found")
    return c