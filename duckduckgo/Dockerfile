FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# $PORT 환경 변수를 동적으로 받도록 수정
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
