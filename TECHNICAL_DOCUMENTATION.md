# Technical documentation

Release `v1.0.0-mlops`. What is implemented, with file references. More detail:
[ARCHITECTURE.md](ARCHITECTURE.md), [MONITORING.md](MONITORING.md), [RETRAINING.md](RETRAINING.md),
[DEPLOYMENT.md](DEPLOYMENT.md), [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md),
[README.md](README.md).

## 1. Overall architecture

A training pipeline (`src/main.py`) produces a versioned model artifact; a file-based registry
(`src/registry.py`) records which version is approved; the FastAPI service (`src/api.py`) serves the
approved version; monitoring (`src/monitoring.py`) observes traffic; gated retraining
(`src/retrain.py`) turns collected feedback into candidates; MLflow (`src/tracking.py`) records
runs and mirrors the approved version with the `champion` alias. Separate runners reuse the same
pipeline for MovieLens 1M (`src/run_ml1m.py`) and MovieTweetings (`src/run_movietweetings.py`).

## 2. Data flow and preprocessing

1. **Download** (`src/data_loader.py`): `ml-latest-small.zip` from files.grouplens.org; TLS with the
   shipped public GroupLens intermediate certificate (`certs/`); SHA-256 `696d65a3…` verified
   before extraction; path-traversal-safe extraction.
2. **Validation**: required columns, types, rating bounds 0.5–5.0 in 0.5 steps, duplicates
   resolved, dataset fingerprint (SHA-256 of the sorted ratings) recorded (`fce385dc…`).
3. **Split** (`train_val_test_split`): per user, stratified, seed 42: 70 % train / 10 % validation /
   20 % test (100,836 ratings → 70,579 / 10,093 / 20,164); zero shared user–movie pairs; split
   checksum recorded; a leakage detector re-checks it.
4. **Matrix**: train ratings → user × item matrix; global mean, user and item biases removed;
   missing entries imputed with 0 (`preprocessing.imputation_strategy`).

## 3. Model, selection and evaluation

* **Model**: truncated SVD of the bias-adjusted matrix (`src/training.py`); prediction = global
  mean + user bias + item bias + `U_k Σ_k V_kᵀ`, clipped to 0.5–5.0. Full SVD for small matrices,
  randomized SVD above a size threshold.
* **Selection**: k ∈ {5, 10, 20, 50, 100, 150, 200}; k with the lowest **validation RMSE** is chosen
  (k = 20, validation RMSE 0.8729). The test split is never seen during selection
  (`tests/test_integration.py`).
* **Final evaluation** (once, on test): RMSE 0.8853, MAE 0.6754 for SVD k = 20; baselines global
  mean, user mean, item mean, user + item bias (0.8943), SGD-MF k = 20 (0.8737), most popular.
* **Ranking** (`src/ranking.py`): for users with test ratings, rated train items masked; relevance
  = held-out rating ≥ 4.0; Precision@10, Recall@10, NDCG@10, hit rate, catalogue coverage; items
  with < 5 training ratings excluded from candidate lists. SVD NDCG@10 0.0109 vs most popular 0.1470.
* **Outputs**: `outputs/corrected/` (metrics JSON/CSV, k sweep, split manifest, plots).
  Historical test-selected results remain in `outputs/` with `HISTORICAL_RESULTS.md`.

## 4. Inference

`src/recommender.py` (`Recommender`): loads `model.npz` after verifying its SHA-256, holds the
factors, biases, id maps, movie titles/genres and popularity list.

* `recommend(user, n, exclude_rated, exclude_movie_ids)`: known user → scores for all eligible
  items, rated and excluded items removed, top-n (`strategy: svd`); unknown user → most popular
  items by training rating count (`strategy: popularity_fallback`).
* `predict_rating(user, movie)`: SVD for known pairs; otherwise user-bias-only, item-bias-only or
  global mean (`method` field).

## 5. API architecture

`src/api.py` (`create_app`): `ServiceState` loads the approved version at start-up and owns the
monitor and the feedback writer. Endpoints: `GET /health`, `GET /model/info`,
`GET /recommendations/{user_id}`, `POST /recommendations`, `POST /predict`, `POST /feedback`,
`GET /metrics`, `GET /monitoring/summary`, `POST /model/reload` ([API_REFERENCE.md](API_REFERENCE.md),
`docs/openapi.json`). Pydantic validation with `extra="forbid"`; ids > 0; `n` ≤ `api.max_n` (100);
ratings 0.5–5.0; uniform error JSON with `request_id`; access log without bodies or user ids;
`/model/reload` disabled unless `SVD_ADMIN_TOKEN` is set (constant-time comparison).

## 6. MLflow integration

`src/tracking.py` (`Tracker`): default `sqlite:///mlflow.db`, artifacts `mlartifacts/`, experiment
`svd-movie-recommender` (MovieTweetings: `svd-movietweetings`). Logged: parameters (k grid,
selected k, split settings), per-k validation metrics, validation/test metrics, dataset and split
fingerprints, git commit, artifacts. Model versions are created from the run's artifact and tagged
`registry_version=<id>`; alias `champion` mirrors the approved registry version (promotion, and
since this delivery also rollback), `rejected-latest` marks the last rejected candidate.

## 7. Model registry

`src/registry.py`: `models/registry.json` (approved pointer, history, events) + `models/versions/<id>/`
(`model.npz`, `metadata.json` with metrics, fingerprints, checksum, git commit). Atomic writes,
lock file, version directories readable by other users (0755) so the API container can serve a
registry read-only. Register ≠ approve: approval and rollback need a reason and are logged.

## 8. Monitoring and drift detection

`src/monitoring.py` (`RecommendationMonitor`): per request it records a **salted SHA-256 pseudonym**
of the user (salt `SVD_MONITORING_SALT`; a random per-process salt if unset or left at the
placeholder), user-activity bucket, strategy, popularity deciles of recommended items, latency.
Signals: Prometheus `recsys_*` metrics; input drift (PSI of user-activity buckets vs training
users); prediction drift (PSI of recommended-item popularity deciles vs the build-time reference,
SVD responses only); unknown-user / cold-start rates; observed RMSE/MAE from `POST /feedback`
(from 30 labels). PSI status ok < 0.10 ≤ warning < 0.25 ≤ alert; `insufficient_data` below 50
requests. Event log `monitoring/events.jsonl`; in-memory state resets on restart.

## 9. Retraining workflow

`python src/retrain.py --feedback FILE.csv [--dry-run]` (CSV `userId,movieId,rating[,timestamp]`, the
format `POST /feedback` writes to `data/feedback/api_feedback.csv`):

1. Load the approved version; reload base data and check its fingerprint; re-create the base split
   and check its checksum. The **test partition is never used** (enforced by a test).
2. Validate feedback; drop invalid rows, duplicates and pairs already in base data or earlier
   promoted feedback. Require ≥ 100 new ratings, ≥ 5 users.
3. Split new ratings deterministically (salted hash) into 80 % feedback-train / 20 %
   feedback-validation.
4. Re-run k selection on base validation + feedback-validation with base train + feedback-train.
5. Compare candidate with the approved model on the same validation data; gate; register; promote
   or reject; JSON report in `reports/retraining/`.

## 10. Promotion and rollback

Gate (`config.yaml` → `retraining.promotion`): candidate RMSE on the combined validation ≤ approved −
`min_rmse_improvement` (0.0); base-validation RMSE may degrade by at most 0.002; NDCG@10 may drop by at
most 0.01. Promotion is **automatic** when the gate passes (not with `--dry-run`). Rollback:
`python src/retrain.py --rollback --reason "..."` re-approves the previous version and moves the MLflow
`champion` alias with it. A running API serves the new approved version after a restart or
`POST /model/reload` with `X-Admin-Token`. **Production note:** with `min_rmse_improvement` 0.0 a
candidate that merely ties the approved model passes (observed on MovieTweetings with random
feedback); set a positive margin and/or a human approval step.

## 11. MovieTweetings pipeline

`src/movietweetings.py` + `src/run_movietweetings.py`: checksum-pinned download of `ratings.dat`,
`movies.dat` and `LICENSE` from the author's GitHub repository at commit `44c525d0…`; `users.dat`
(Twitter account ids) never downloaded; ratings of 0 dropped; ratings / 2 (1–10 → 0.5–5.0);
iterative 20-core filter; same split/selection/evaluation code; SGD-MF disabled; own registry
`models_external/movietweetings/`, outputs `outputs/external/movietweetings/`, MLflow experiment
`svd-movietweetings`; `--retrain-feedback`, `--dry-run`, `--rollback`. Results:
[docs/MOVIETWEETINGS_EVALUATION.md](docs/MOVIETWEETINGS_EVALUATION.md).

## 12. Configuration and environment variables

All settings in `config.yaml` (or `$SVD_CONFIG`); environment variables override:

| Variable | Default | Purpose |
|---|---|---|
| `SVD_MODEL_DIR` | `models` | registry location |
| `SVD_DATA_DIR`, `SVD_RATINGS_PATH` | `data`, `data/ml-latest-small/ratings.csv` | dataset location |
| `SVD_OUTPUT_DIR` | `outputs/corrected` | pipeline outputs |
| `SVD_MONITORING_DIR` | `monitoring` | event log |
| `SVD_MONITORING_SALT` | unset → random per process | pseudonymisation salt (**secret**) |
| `SVD_FEEDBACK_PATH` | `data/feedback/api_feedback.csv` | where `/feedback` appends ratings |
| `SVD_RETRAINING_REPORT_DIR` | `reports/retraining` | retraining reports |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `SVD_MLFLOW_ENABLED`, `SVD_MLFLOW_ARTIFACTS` | `true`, `mlartifacts` | tracking switch, artifact location |
| `SVD_API_MAX_N` | `100` | maximum `n` |
| `SVD_ADMIN_TOKEN` | unset | enables `/model/reload` (**secret**) |
| `SVD_LOG_LEVEL`, `LOG_FORMAT` | `INFO`, `json` (API) | logging |
| `API_BIND`, `API_PORT` | `127.0.0.1`, `8000` | Compose port publishing |
| `SVD_API_NO_AUTOCREATE` | – | `1` stops the module building an app at import (tests) |
| `GIT_COMMIT` | – | commit recorded when `.git` is absent |

## 13. Docker architecture

`Dockerfile` targets: `base` (API; `requirements-api.txt`; user `app` uid 10001; only `models/`,
`monitoring/`, `data/` writable; health check on `/health`) and `trainer` (adds the training stack,
MLflow, scripts). pip/setuptools of the base image are upgraded; optional BuildKit secret `pip_ca`.
`docker-compose.yml`: `api` (port `127.0.0.1:8000`, volumes `models` **read-only**, `monitoring`,
`feedback`; `cap_drop: ALL`; `no-new-privileges`; 1 GB memory limit), `trainer` (profile `tools`;
writes `models`, `dataset`, `mlflow-data`), `mlflow` (profile `mlflow`; official
`ghcr.io/mlflow/mlflow:v3.18.0` on `127.0.0.1:5000`).

## 14. CI/CD workflow

`.github/workflows/ci.yml` on pull requests, pushes to `main` (and the historical delivery branch
name) and manual dispatch:

1. **Lint + tests** on Python 3.12 and 3.13 (ruff; pytest with synthetic fixtures, no network).
2. **Docker**: build the API image, build a demo model on synthetic data, start the container,
   call the API.
3. **Full pipeline on MovieLens latest-small**: fresh download, checksum, train, select on
   validation, evaluate on test; assert zero leakage, validation-based selection and the dataset
   checksum.

No automatic deployment; releases are tagged manually.

## 15. Security considerations

[SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md) and [SECURITY.md](SECURITY.md): no
authentication/TLS/rate limiting in the application (gateway prerequisites); localhost-only
Compose; non-root, capability-dropped container with read-only registry; pseudonymised monitoring;
`/feedback` stores raw user ids and ratings (personal data, needs a retention policy).

## 16. Known technical limitations

* Dataset licence (MovieLens research-only; MovieTweetings legal review).
* Popularity beats SVD on offline top-10 metrics; random (not temporal) split.
* Dense user × item matrix: memory grows with users × items (MovieTweetings needed a 20-core filter).
* Monitoring state per process; no horizontal scaling.
* Permissive default promotion gate (§10).
