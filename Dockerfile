# Production Dockerfile for Verace V1 Architecture
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml /app/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir uvicorn fastapi pytest

COPY . /app
RUN pip3 install --no-cache-dir -e .

EXPOSE 8000

CMD ["python3", "-m", "verace_v1.serving.api"]
