FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .

# pip 최신화 및 캐시를 완전히 무시하고 패키지 새로 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
