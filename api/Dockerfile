# ────────────────────────────────────────────────────────────────── builder
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY api/requirements.txt /build/requirements.txt
RUN pip install --user --no-cache-dir -r /build/requirements.txt \
    && python -m nltk.downloader -d /root/nltk_data stopwords

# ────────────────────────────────────────────────────────────────── runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/api/.local/bin:${PATH}" \
    PYTHONPATH="/app"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 api

USER api
WORKDIR /app

COPY --from=builder --chown=api:api /root/.local /home/api/.local
COPY --from=builder --chown=api:api /root/nltk_data /home/api/nltk_data

ENV NLTK_DATA=/home/api/nltk_data

COPY --chown=api:api api/ /app/api/
COPY --chown=api:api src/ /app/src/
COPY --chown=api:api config/ /app/config/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
