# Screwed-Backend
=======
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

```bash
uvicorn app.main:app --reload
```

전체 프로젝트
- NASA에서 기본 제공: 26개 변수로 이루어진 기후 데이터
- 국가별 농장물 생산량(output)
- 변수가 조금 더 다양하긴 하는데
- 모델 자체는 
    예측값 만들기. 
    LSTM 만들기
    - 
    파일; 영향 받아서. 
    -> 태양 에너지 등등. 
    작물별; 상관관계;

게임 흐름으로 만들면 -> 삼국지 한것처럼. 
경작하는 게임. 
상황별 사용자가 데이터 입력. 
결과값 예측. 
- 
결과값, 입력값 -> 상황같은것을 Gemini가 만들어주는것. 
도현이: 분기 어쩌고저쩌고. 

우리가 줘야 하는 것: 
    NaN
    지역별로 옥수수, 쌀, 콩, 밀 -> 재배되는 계절에 따라서 
    순서가 반대여야 한다. 

- Unity(프론트, 게임 클라이언트)
    - 게임 루프/연출: 시즌(봄, 여름, 가을 겨울), 월 단위 턴 진행, 중간중간 이벤트, 자원, 생산량 등
    입력 수집/검증: 플레이어가 지정한 지역, 작물(옥수수/쌀/콩/밀), 경작면적, 관개·비료 정책, 병해충 방제 등.
	•	시각화: 기후 타임라인(온도/강수), 수확량 예측 그래프, 불확실성(신뢰구간) 표시.
	•	로컬 캐시/오프라인: 최근 호출 결과 로컬 저장(끊김 시 임시 플레이).
	•	경량 규칙 계산: 아주 단순한 즉시 피드백(예: 입력 범위 체크, 기본 비용 계산)만 로컬에서.


-> API 명세 관리해야함. 

- FastAPI:
    - 데이터 레이어
        - NASA 기후 26변수 인제스트/업데이트, 국가, 지역별 농작물 실제 산출량(ground thruth):
    - 지형 생성 알고리즘; 
        - GET /metadata
            - 지역 목록, 반구 정보, 작물별 재배 달력(파종/수확 윈도우), 사용 가능한 기후 변수 
- 지도 데이터에 대해, 지도 데이터가 있는 것이 좋을까/
        - POST /scenario
        - POST /predict
        - POST /save
