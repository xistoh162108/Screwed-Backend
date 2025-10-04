from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import IntegrityError
from typing import Optional, List
import uuid
from datetime import datetime, timezone

from app.models import Session as SessionModel
from app.schemas import SessionCreate, SessionUpdate, SessionOut

def now_iso():
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _new_id(prefix: str = "s") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def to_out(obj: SessionModel) -> SessionOut:
    return SessionOut(
        id=obj.id,
        title=obj.title,
        root_turn_id=obj.root_turn_id,
        meta=obj.meta or {},
        created_at=obj.created_at,
        updated_at=obj.updated_at,
    )

def create(db: DBSession, body: SessionCreate) -> SessionOut:
    sid = _new_id()
    now = now_iso()
    obj = SessionModel(
        id=sid,
        title=body.title,
        root_turn_id=body.root_turn_id,
        meta=body.meta or {},
        created_at=now,
        updated_at=now,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise ValueError("Failed to create Session: " + str(e)) from e
    db.refresh(obj)
    return to_out(obj)

def get(db: DBSession, session_id: str) -> Optional[SessionOut]:
    obj = db.get(SessionModel, session_id)
    return to_out(obj) if obj else None

def list_(db: DBSession, limit: int = 50, cursor: Optional[str] = None) -> List[SessionOut]:
    q = db.query(SessionModel)
    if cursor:
        q = q.filter(SessionModel.id > cursor)
    q = q.order_by(SessionModel.id).limit(min(limit, 200))
    return [to_out(o) for o in q.all()]

def update(db: DBSession, session_id: str, body: SessionUpdate) -> Optional[SessionOut]:
    obj = db.get(SessionModel, session_id)
    if not obj:
        return None
    if body.title is not None:
        obj.title = body.title
    if body.root_turn_id is not None:
        obj.root_turn_id = body.root_turn_id
    if body.meta is not None:
        obj.meta = body.meta
    obj.updated_at = now_iso()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return to_out(obj)

def delete(db: DBSession, session_id: str) -> bool:
    obj = db.get(SessionModel, session_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True