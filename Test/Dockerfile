# Python 3.10 슬림 버전 사용 (가볍고 빠름)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 빌드에 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y build-essential

# 요구사항 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 코드 복사
COPY . .

# 8080 포트 노출 (Cloud Run 기본 포트)
EXPOSE 8080

# [실전 에러 방지 조건 2] Docker 실행 파라미터 정확히 지정 (0.0.0.0 바인딩)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
