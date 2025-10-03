import os
from app.db.base import Base, make_engine
from app.models.user import User  # 모델을 로드해야 테이블 생성 대상에 포함됨

def main():
    uri = os.environ["APP_SQLALCHEMY_URI"]
    engine = make_engine(uri)
    Base.metadata.create_all(bind=engine)
    print("테이블 생성 완료")

if __name__ == "__main__":
    main()