from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class SessionBase(BaseModel):
    title: Optional[str] = None
    root_turn_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class SessionCreate(SessionBase):
    pass  # 생성 시 필수값이 없다면 그대로

class SessionUpdate(SessionBase):
    pass

class SessionOut(SessionBase):
    id: str
    created_at: datetime
    updated_at: datetime