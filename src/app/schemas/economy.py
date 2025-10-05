# app/schemas/economy.py
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
from decimal import Decimal

class SessionEconomyBase(BaseModel):
    currency: Literal["USD", "EUR", "KRW", "JPY", "CNY"] = "USD"
    balance: Decimal = Field(..., max_digits=18, decimal_places=2)

class SessionEconomyCreate(SessionEconomyBase):
    pass

class SessionEconomyUpdate(BaseModel):
    currency: Literal["USD", "EUR", "KRW", "JPY", "CNY"] | None = None
    balance: Decimal | None = Field(None, max_digits=18, decimal_places=2)

class SessionEconomyOut(SessionEconomyBase):
    id: str
    session_id: str
    updated_at: datetime

    class Config:
        from_attributes = True
