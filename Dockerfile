FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

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

COPY app.py .

EXPOSE 5001

CMD ["python", "app.py"]