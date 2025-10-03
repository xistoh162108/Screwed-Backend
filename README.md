# Screwed-Backend

# Readme

## Directory
```
from pydantic import BaseModel, EmailStr

# 공통 속성 (요청과 응답에서 같이 쓰이는 부분)
class UserBase(BaseModel):
    email: EmailStr   # 이메일 형태 자동 검사 (aaa@bbb.com 아니면 에러)
    name: str         # 사용자 이름

# 생성(Create) 요청 전용 스키마
class UserCreate(UserBase):
    pass   # UserBase 그대로 사용 (추가 필드 필요하면 여기에 씀)

# 읽기(Read) 응답 전용 스키마
class UserRead(UserBase):
    id: int           # DB에 저장된 고유 ID
    class Config:
        from_attributes = True   # SQLAlchemy 모델 → Pydantic 변환 가능하게 함
```