from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
from app.schemas.output import OutputCreateIn, OutputIdOut, OutputOut
from app.services import output_service

router = APIRouter(tags=["outputs"])

# Command -> Output 생성 (멱등: 이미 있으면 기존 id 반환)
@router.post("/commands/{command_id}/outputs", response_model=OutputIdOut, status_code=201)
def create_output_from_command(command_id: str, body: Optional[OutputCreateIn] = None, db: Session = Depends(get_db)):
    try:
        return output_service.create_from_command(db, command_id, body)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# 특정 Output 조회
@router.get("/outputs/{output_id}", response_model=OutputOut)
def get_output(output_id: str, db: Session = Depends(get_db)):
    out = output_service.get(db, output_id)
    if not out:
        raise HTTPException(status_code=404, detail="Output not found")
    return out

# 특정 Turn의 Outputs 목록 (옵션)
@router.get("/turns/{turn_id}/outputs", response_model=List[OutputOut])
def list_outputs_by_turn(
    turn_id: str,
    limit: int = Query(100, ge=1, le=500),
    cursor: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return output_service.list_by_turn(db, turn_id, limit, cursor)

# Output 적용(커밋): PROCESSED & COMPLETE만 허용
@router.post("/outputs/{output_id}/apply")
def apply_output(output_id: str, db: Session = Depends(get_db)):
    try:
        return output_service.apply(db, output_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))