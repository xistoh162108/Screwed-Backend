from app.db.base import Base, engine
from app.models.user import User

def init_db():
    # 모든 모델 테이블 생성
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("DB 초기화 완료")