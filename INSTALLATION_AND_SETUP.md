# Installation and setup

Release `v1.0.0-mlops`. Each block is marked:

* **[Tested]**: executed in the final delivery QA ([FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md)).
* **[Executed earlier]**: executed during the delivery (evidence in `docs/evidence/`).
* **[Not independently verified]**: documented for convenience; not run by us.

## 1. Supported environments

| Item | Status |
|---|---|
| Linux x86-64 (Ubuntu 24.04), Python 3.12.3 | **verified** |
| Python 3.13 | **verified in CI** (lint + tests) |
| Docker 24+ / Compose v2, image base Debian 13 (`python:3.12-slim`) | **verified** |
| macOS, Windows (native / WSL 2) | not verified; expected to work with Python 3.12+ or Docker |
| Python 3.11 or older | not supported |

Hardware: training ≈ 1.5 GB RAM (latest-small), ≈ 3 GB for MovieLens 1M; API ≈ 300 MB. No GPU.

## 2. Get the code **[Tested]**

```bash
git clone https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering.git
cd SVD-Based-Collaborative-Filtering
git checkout v1.0.0-mlops          # the released version (or stay on main)
```

From the release ZIP: unzip it and `cd` into the extracted folder.

## 3. Virtual environment and dependencies **[Tested]**

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1  [Not independently verified]
pip install -r requirements.txt -r requirements-dev.txt
pip check
```

`requirements.txt`: training + MLOps (pinned). `requirements-api.txt`: what the API image installs.
`requirements-dev.txt`: tests and lint.

## 4. Configuration

```bash
cp .env.example .env               # optional
python -c "import secrets; print(secrets.token_hex(16))"   # put the output in SVD_MONITORING_SALT
```

Settings: `config.yaml`; overrides via environment variables
([TECHNICAL_DOCUMENTATION.md §12](TECHNICAL_DOCUMENTATION.md#12-configuration-and-environment-variables)).
Secrets: `SVD_MONITORING_SALT` (pseudonymisation) and `SVD_ADMIN_TOKEN` (enables `/model/reload`).
Never commit `.env`.

## 5. Train and approve the model **[Tested]**

```bash
python src/main.py --approve --approval-reason "reviewed validation results"
```

Downloads MovieLens latest-small (checksum-verified, ≈ 1 MB), trains, selects k on validation,
evaluates on test once, writes `outputs/corrected/`, registers and approves the version in
`models/`, logs to `mlflow.db`. Takes ≈ 30 s. Expected: `k=20 | val RMSE 0.8729 | test RMSE 0.8853`,
version `svd-k20-fce385dc-c6b9d577`. Without `--approve` the version is registered but not served.

Behind a TLS-inspecting proxy **[Tested]** (in the delivery environment):
`SSL_CERT_FILE=/path/to/proxy-ca.pem python src/main.py ...`. Without internet access, copy a
verified `ml-latest-small.zip` into `data/`; its SHA-256 is checked before use.

Optional **[Executed earlier]**: `python src/run_ml1m.py` (MovieLens 1M cross-check,
`docs/evidence/08_ml1m_run.log`); `python src/run_movietweetings.py` (alternative dataset,
`docs/evidence/movietweetings/`).

## 6. Start the API locally **[Tested]**

```bash
uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health          # {"status":"ok","model_loaded":true,...}
./examples/demo/run_demo.sh
```

Swagger UI: http://127.0.0.1:8000/docs. Stop with Ctrl+C.

## 7. Docker **[Tested]**

```bash
docker compose config -q                               # validate
docker compose build api
docker compose --profile tools build trainer
docker compose --profile tools run --rm trainer        # download, train, evaluate, approve -> 'models' volume
docker compose up -d api                               # http://127.0.0.1:8000
docker compose ps                                      # (healthy) once the model is loaded
curl http://127.0.0.1:8000/health
docker compose logs -f api
docker compose restart api                             # model, events and feedback persist (named volumes)
```

* The API mounts the `models` volume read-only and runs as uid 10001 with all capabilities dropped.
* The port is published on 127.0.0.1 only; set `API_BIND=0.0.0.0` only behind a TLS proxy with
  authentication.
* Behind a TLS-inspecting proxy: build with `DOCKER_BUILDKIT=1 docker build --secret
  id=pip_ca,src=/path/to/proxy-ca.pem ...`; for the trainer's dataset download mount the CA and set
  `SSL_CERT_FILE` (see DEPLOYMENT.md). In the delivery environment the proxy listened on localhost,
  so the trainer was run with host networking (`docker run --network host ...` on the same
  volumes); this is specific to that environment.
* Serving the MovieTweetings model instead: see DEPLOYMENT.md (only after legal review).

## 8. MLflow

Local server against the project database **[Tested]**:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts
# http://127.0.0.1:5000 (no authentication: keep it on localhost)
```

Containerised server for the Compose volumes: `docker compose --profile mlflow up -d mlflow`
(status in the QA report; it needs access to ghcr.io).

## 9. Health checks

| Check | Expected |
|---|---|
| `curl http://127.0.0.1:8000/health` | HTTP 200, `"model_loaded": true` |
| no approved model | HTTP 503, `"status": "unavailable"`, `detail` gives the reason |
| `docker compose ps` | `(healthy)` (image health check every 15 s, 20 s start period) |

## 10. Stopping services and cleaning up **[Tested]**

```bash
docker compose down                    # stop, keep volumes (model, monitoring, feedback, dataset, MLflow)
docker compose down -v                 # also delete all named volumes
deactivate
rm -rf .venv data/ml-latest-small data/*.zip monitoring reports mlartifacts mlflow.db   # local state
```

`models/` holds the registry; delete it only to remove all model versions.
