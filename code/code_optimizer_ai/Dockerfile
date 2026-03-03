# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m venv "${VENV_PATH}" && \
    pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install -r requirements.txt && \
    find "${VENV_PATH}" -type d -name '__pycache__' -exec rm -rf {} + && \
    find "${VENV_PATH}" -type f -name '*.pyc' -delete && \
    find "${VENV_PATH}" -type f -name '*.pyo' -delete


FROM ${PYTHON_IMAGE} AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    libpq5 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"
COPY --chown=appuser:appuser . .

RUN install -d -o appuser -g appuser /app/data /app/models /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "code.code_optimizer_ai.run"]


FROM production AS development

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
USER appuser

CMD ["uvicorn", "code.code_optimizer_ai.api_main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]
