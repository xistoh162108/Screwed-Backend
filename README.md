# Screwed-Backend

# Readme

## Directory
```
your-app/
├─ app/
│  ├─ main.py                 # 앱 엔트리포인트
│  ├─ api/
│  │  ├─ deps.py              # 공용 의존성(예: DB 세션)
│  │  └─ v1/
│  │     ├─ __init__.py
│  │     └─ endpoints/
│  │        ├─ __init__.py
│  │        ├─ health.py      # /health, /ping
│  │        └─ users.py       # 예시 CRUD
│  ├─ core/
│  │  └─ config.py            # 설정(.env 읽기)
│  ├─ db/
│  │  ├─ base.py              # Base/엔진/세션
│  │  └─ init_db.py           # 초기 데이터(Optional)
│  ├─ models/
│  │  └─ user.py              # SQLAlchemy 모델
│  └─ schemas/
│     └─ user.py              # Pydantic 스키마(입출력)
├─ tests/
│  └─ test_health.py
├─ .env.example
├─ requirements.txt
├─ Dockerfile
└─ README.md
```