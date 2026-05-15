FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY app/ app/

RUN pip install --no-cache-dir -e .

# Giữ nguyên EXPOSE để báo hiệu cổng mặc định cho local
EXPOSE 8001

# Sửa dòng CMD: Dùng shell (-c) để đọc được biến môi trường $PORT. 
# ${PORT:-8001} nghĩa là: Nếu có biến PORT thì dùng, nếu không (như ở local) thì chạy cổng 8001
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]