# Production Dockerfile for Verace V1 Architecture
#
# Multi-stage: the builder stage installs into an isolated venv (needs git for
# pip's VCS support); the runtime stage copies only that venv and the app source,
# with no git/pip cache/build tools, and runs as a non-root user.

# ---- Builder ----
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn fastapi

COPY . /app
RUN pip install --no-cache-dir -e .

# ---- Runtime ----
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 verace \
    && useradd --uid 1000 --gid verace --shell /bin/bash --create-home verace

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
USER verace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["python3", "-m", "verace_v1.serving.api"]
