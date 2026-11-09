FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY assets/ /app/assets/
COPY README.md /app/

# Tạo sẵn thư mục rỗng để mount volume
RUN mkdir -p /app/weights /app/runs

# Chạy ảnh kiểm thử có sẵn trong assets làm mặc định
CMD ["python", "src/image_detection.py", "--source", "assets/BoxPR_curve.png", "--weights", "weights/best.pt"]