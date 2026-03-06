FROM python:3.10-slim

# Python 환경 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# YOLO / OpenCV 관련 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사 (Docker cache 최적화)
COPY requirements.txt .

# pip 업데이트 및 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 서버 코드 및 모델 복사
COPY . .

# YOLO 모델 경로
ENV YOLO_WEIGHTS_PATH=/app/best.pt

# MCP 서버 포트
ENV PORT=8001

EXPOSE 8001

CMD ["python", "server.py"]