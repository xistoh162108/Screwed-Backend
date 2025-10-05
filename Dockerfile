# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (빌드에 필요한 기본 도구만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PATH="/usr/local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# 기본 엔트리포인트는 uvicorn (docker-compose에서 worker는 명령만 바꿔줌)
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]