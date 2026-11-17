# syntax=docker/dockerfile:1
# SVD movie recommender API: serving image.
# Model artifacts are NOT baked in: mount the registry at /app/models
# (produced by `python src/main.py --approve` or `docker compose run --rm trainer`).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    SVD_MODEL_DIR=/app/models \
    SVD_MONITORING_DIR=/app/monitoring \
    SVD_FEEDBACK_PATH=/app/data/feedback/api_feedback.csv \
    SVD_MLFLOW_ENABLED=false

WORKDIR /app

# The base image ships pip/setuptools versions with published advisories (pip-audit,
# 2026-11-17); upgrade them before anything else is installed.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --upgrade pip==26.3 setuptools==84.2.1

COPY requirements-api.txt ./
# Optional: behind a TLS-inspecting proxy, pass its CA bundle as a build secret:
#   docker build --secret id=pip_ca,src=/path/to/ca.pem .
# Without the secret this is a plain `pip install`. The CA never enters an image layer.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements-api.txt

# Unprivileged runtime user
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app

COPY config.yaml ./
COPY src/ ./src/
# Only the state directories are writable by the runtime user; code and config stay root-owned.
RUN mkdir -p /app/models /app/monitoring /app/data/feedback && chown -R app:app /app/models /app/monitoring /app/data

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Training image: full dependencies, used by `docker compose run trainer` ──
FROM base AS trainer
USER root
COPY requirements.txt ./
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements.txt
COPY certs/ ./certs/
COPY scripts/ ./scripts/
ENV SVD_MLFLOW_ENABLED=true \
    MLFLOW_TRACKING_URI=sqlite:////mlflow-data/mlflow.db \
    SVD_MLFLOW_ARTIFACTS=/mlflow-data/mlartifacts \
    MLFLOW_DISABLE_AGENT_HINT=1
RUN mkdir -p /app/data /app/outputs /app/reports /mlflow-data && chown -R app:app /app /mlflow-data
USER app
CMD ["python", "src/main.py", "--approve", "--no-plots"]
