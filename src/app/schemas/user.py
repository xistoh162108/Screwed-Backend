from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    is_active: bool =