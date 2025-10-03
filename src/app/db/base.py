from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

Base = declarative_base()

def make_engine(sqlalchemy_uri: str):
    # Cloud SQL은 연결이 간헐적으로 끊길 수 있어 pool_pre_ping 권장
    return create_engine(sqlalchemy_uri, pool_pre_ping=True, future=True)

def make_session_factory(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)