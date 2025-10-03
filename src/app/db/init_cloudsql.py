import os
import psycopg  # psycopg3
from contextlib import closing

def admin_conn():
    return psycopg.connect(
        host=os.environ["ADMIN_PGHOST"],
        port=os.environ.get("ADMIN_PGPORT", "5432"),
        user=os.environ["ADMIN_PGUSER"],
        password=os.environ["ADMIN_PGPASSWORD"],
        dbname=os.environ.get("ADMIN_DB", "postgres"),
        autocommit=True,  # DDL 편하게
    )

def role_exists(cur, role):
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    return cur.fetchone() is not None

def db_exists(cur, dbname):
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    return cur.fetchone() is not None

def create_role_if_needed(cur, role, pwd):
    if not role_exists(cur, role):
        cur.execute(f"CREATE ROLE {psycopg.sql.Identifier(role).as_string(cur)} LOGIN PASSWORD %s", (pwd,))
        # 필요한 경우 CREATEDB 등 권한 부여 가능
        # cur.execute(f"ALTER ROLE {role} CREATEDB")

def create_db_if_needed(cur, dbname, owner):
    if not db_exists(cur, dbname):
        # OWNER 지정하여 생성
        q = f"CREATE DATABASE {psycopg.sql.Identifier(dbname).as_string(cur)} OWNER {psycopg.sql.Identifier(owner).as_string(cur)}"
        cur.execute(q)

def grant_schema_defaults(db_conn, role):
    with db_conn.cursor() as cur:
        # public 스키마 권한(필요 시 조정)
        cur.execute("GRANT USAGE, CREATE ON SCHEMA public TO {}".format(psycopg.sql.Identifier(role).as_string(cur)))
        # 이후 생성되는 객체 기본 권한
        cur.execute("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {};
        """.format(psycopg.sql.Identifier(role).as_string(cur)))
        cur.execute("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {};
        """.format(psycopg.sql.Identifier(role).as_string(cur)))

def main():
    app_db     = os.environ["APP_DB_NAME"]
    app_user   = os.environ["APP_DB_USER"]
    app_pass   = os.environ["APP_DB_PASSWORD"]

    with closing(admin_conn()) as conn, conn.cursor() as cur:
        create_role_if_needed(cur, app_user, app_pass)
        create_db_if_needed(cur, app_db, app_user)

    # 스키마 기본 권한 설정을 위해 새로 만든 DB로 접속
    with closing(psycopg.connect(
        host=os.environ["ADMIN_PGHOST"],
        port=os.environ.get("ADMIN_PGPORT", "5432"),
        user=os.environ["ADMIN_PGUSER"],
        password=os.environ["ADMIN_PGPASSWORD"],
        dbname=app_db,
        autocommit=True,
    )) as appdb_conn:
        grant_schema_defaults(appdb_conn, app_user)

    print("Cloud SQL 초기화 완료: 역할·DB·스키마 권한 설정 끝")

if __name__ == "__main__":
    main()