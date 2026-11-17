# Deployment

Everything here was executed in the acceptance run (logs in `docs/evidence/03_*`, `04_*`).
**No public cloud deployment has been performed**; this is a local / single-host
container deployment.

## Requirements

| | API | Trainer |
|---|---|---|
| CPU | 1 core | 1–2 cores |
| RAM | ~300 MB in use; 1 GB limit set in compose | ~1.5 GB (latest-small); ~3 GB for ML-1M |
| GPU | not used | not used |
| Disk | image 396 MB + model ≈ 2 MB | image 1.28 GB + dataset 3 MB + MLflow |

## Images

`Dockerfile` has two targets: `base` (API, `requirements-api.txt`, non-root user
`app` uid 10001, health check on `/health`, port 8000) and `trainer` (adds training,
MLflow and scripts). No secrets or data are copied into images (`.dockerignore`).

## Commands

```bash
cp .env.example .env                                   # optional, edit values
docker compose build api                               # build API image
docker compose --profile tools build trainer           # build trainer image
docker compose --profile tools run --rm trainer        # download, train, evaluate, approve -> 'models' volume
docker compose up -d api                               # start (restart: unless-stopped)
docker compose ps                                      # shows (healthy) once the model is loaded
docker compose logs -f api                             # JSON logs
docker compose restart api                             # state survives (named volumes)
docker compose down                                    # stop; volumes are kept
docker compose down -v                                 # stop and DELETE volumes (models, logs, feedback)
docker compose build api && docker compose up -d api   # rebuild after code changes
```

Tests are not shipped inside the images. Run them on the host (`pytest -q`) or rely on
CI, which also builds the image and calls the API inside a running container.

Optional MLflow UI: `docker compose --profile mlflow up -d mlflow` → http://localhost:5000
(reads the same `mlflow-data` volume as the trainer).

Retraining in the container:

```bash
docker compose --profile tools run --rm trainer python src/retrain.py --feedback /app/data/feedback/api_feedback.csv
# then, with SVD_ADMIN_TOKEN set in .env, switch the running API to the new approved version:
curl -X POST -H "X-Admin-Token: $SVD_ADMIN_TOKEN" http://localhost:8000/model/reload
```

## Volumes (persist across restarts and rebuilds)

`models` (registry), `monitoring` (event log), `feedback` (POST /feedback),
`dataset` (MovieLens archive), `mlflow-data` (SQLite + artifacts). Verified: after
`docker compose restart api` the same model version is served and earlier events
and feedback are still present.

## Networks with a TLS-inspecting proxy

* Build: `docker build --secret id=pip_ca,src=/path/to/proxy-ca.pem --target base -t svd-recommender-api:1.0.0 .`
  (the CA is used only during `pip install` and never stored in a layer).
* Dataset download inside the trainer: mount the proxy CA and set `SSL_CERT_FILE`, e.g.
  `docker compose --profile tools run --rm -v /path/ca.pem:/etc/ssl/certs/proxy-ca.pem:ro -e SSL_CERT_FILE=/etc/ssl/certs/proxy-ca.pem trainer`.
* If outbound access to GroupLens is blocked, copy a verified `ml-latest-small.zip`
  into the `dataset` volume; the pipeline re-checks its SHA-256 before use.

## Production hardening not included

TLS termination, authentication/authorisation on the API, horizontal scaling (state
is per process), an external metrics store (Prometheus/Grafana), backups of volumes,
and a commercial licence for MovieLens.
