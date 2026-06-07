# RLPE — Radiolarian Literature Plate Extractor
#
# Multi-stage build:
#   1. builder:  install full dep set (compiles numpy/opencv/scikit-image wheels)
#   2. runtime:  slim image with only the libs we link against at runtime
#
# Build:
#   docker build -t rlpe:dev .
#
# Run the API service:
#   docker run --rm -p 8000:8000 rlpe:dev api
#
# Run a single paper via the CLI (mount your PDF in):
#   docker run --rm -v "$PWD/work:/app/work" -v "$PWD/data:/app/data" \
#     rlpe:dev run --pdf-dir /app/work/pdfs --work-dir /app/work/run --use-opendataloader
#
# Run the eval:
#   docker run --rm -v "$PWD:/app" rlpe:dev eval \
#     --pred /app/work/combined_8_v13_FINAL.jsonl \
#     --gold /app/data/gold/ \
#     --output /app/work/eval.json

# ---- 1. builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for opencv + scikit-image runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# Install the package + the optional deps that are part of the
# production surface (API service, schema validation, OpenDataLoader).
# Heavy ML extras (gemma, sam, train) are NOT installed by default;
# use the `-full` target below for those.
RUN pip install --upgrade pip && \
    pip install .[service,schema,opendataloader]

# ---- 2. runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Same system libs as builder; no compiler toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash rlpe

WORKDIR /app
COPY --from=builder /app/src ./src
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY pyproject.toml README.md ./
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests
COPY work/combined_8_v13_FINAL.jsonl ./work/combined_8_v13_FINAL.jsonl

USER rlpe
VOLUME ["/app/work", "/app/data"]

# Default command: start the API service. Override with `docker run ... rlpe:dev <cmd>`.
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "rlpe.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
