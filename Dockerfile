## base
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch 관련 패키지를 제외하고 설치
RUN pip install --upgrade pip && \
    grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY ./app .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]