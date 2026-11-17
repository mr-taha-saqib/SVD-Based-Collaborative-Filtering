# SVD Collaborative Filtering Movie Recommender: client README

**Start here.** Release `v1.0.0-mlops`; repository
https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering.

> **Important:** the delivered model is trained on **MovieLens**, which is licensed for
> **research use only**. Commercial use needs written permission from GroupLens / the University
> of Minnesota or a commercially licensed replacement dataset. Offline, a simple most-popular list
> beats the SVD model on top-10 ranking metrics. See [Limitations](#11-important-limitations).

## 1. Business problem

Recommending items (here: movies) that a user is likely to rate highly, based on the ratings of
many users (collaborative filtering), and predicting how a user would rate a specific item.

## 2. What the system does

* Learns latent user and movie factors with **truncated SVD** of the bias-adjusted rating matrix.
* Selects the rank `k` on a **validation** split only and evaluates the **test** split once
  (the original research code selected `k` on the test set; this was corrected and the historical
  results are kept, clearly labelled, in `outputs/HISTORICAL_RESULTS.md`).
* Compares against global/user/item mean, user + item bias, SGD matrix factorisation and a
  **most-popular** baseline, with rating metrics (RMSE, MAE) and ranking metrics (Precision@10,
  Recall@10, NDCG@10, hit rate, catalogue coverage).
* Serves recommendations and rating predictions through a **FastAPI** service with a popularity
  fallback for unknown users (cold start).
* Tracks runs in **MLflow**, keeps versions in a **model registry** with approval and rollback,
  monitors traffic and **drift** with pseudonymised user ids, and supports **gated retraining**
  from collected feedback.
* Includes a separate **MovieTweetings** pipeline (alternative dataset, MIT-licensed by its author,
  legal review still required).
* Ships Docker images, Compose and GitHub Actions CI.

## 3. Rating prediction vs top-N recommendation

These are different tasks and the model performs differently on them:

| Task | Metric | SVD (test) | Best simple baseline (test) | Verdict |
|---|---|---|---|---|
| Predict the rating a user gives a movie | RMSE (lower is better) | **0.8853** | user + item bias 0.8943 | SVD slightly better |
| Rank the 10 best movies for a user | NDCG@10 (higher is better) | 0.0109 | **most popular 0.1470** | popularity much better |

So the SVD model is a reasonable **rating predictor**, but its **top-10 lists** are weaker than
simply recommending popular movies in this offline evaluation. Validate rankings with real users
(A/B test) before relying on them.

## 4. Technology stack

Python 3.12 (CI also 3.13), numpy, scipy, pandas, scikit-learn, FastAPI 0.163, Uvicorn,
Pydantic 2, MLflow 3.18, prometheus-client, matplotlib, Docker 24+ / Compose v2, GitHub Actions,
pytest, ruff. No GPU needed.

## 5. Architecture overview

```
MovieLens (download, SHA-256 verified) ──> validation, fingerprint ──> per-user split 70/10/20
   ──> SVD for each k ──> choose k on VALIDATION RMSE ──> test once ──> baselines + ranking metrics
   ──> model artifact ──> models/ registry (approve explicitly) ──> MLflow (alias "champion")
client ──HTTP──> FastAPI (src/api.py) ──> Recommender ──> top-N / predicted rating / fallback
                    └──> monitoring (pseudonymised): metrics, drift, feedback RMSE
POST /feedback ──> data/feedback/api_feedback.csv ──> src/retrain.py ──> gate ──> registry
```

Details: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Repository structure (main parts)

```
src/                pipeline (main.py), data_loader, training, evaluation, ranking, recommender,
                    registry, tracking, monitoring, retrain, api, MovieLens-1M and MovieTweetings runners
tests/              156-test pytest suite + test_pipeline.py (original suite, strengthened)
scripts/            demo model on synthetic data, traffic simulator, synthetic feedback
config.yaml         all settings (split, k grid, ranking, registry, MLflow, monitoring, retraining, API)
certs/              public GroupLens intermediate certificate (dataset download)
outputs/            corrected results (outputs/corrected/), historical results (labelled), MovieTweetings results
examples/demo/      API demo script with captured output
docs/               evidence logs, release notes, OpenAPI schema, dataset licensing, MovieTweetings evaluation
report/, presentation/  original research report and slides
Dockerfile, docker-compose.yml, requirements*.txt, pyproject.toml, .env.example
```

MovieLens data is **not** included (licence); it is downloaded and checksum-verified on first run.

## 7. Requirements

* Python **3.12** or 3.13; verified on Linux x86-64 (Ubuntu 24.04) and in Debian-based containers.
* 1–2 CPU cores, ≈ 1.5 GB RAM for training on latest-small, ≈ 300 MB for the API.
* Internet access to `files.grouplens.org` on the first run. No API keys needed.
* Docker 24+ with Compose v2 for the container route.

## 8. Installation and start (short version)

Full, tested instructions: [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md).

```bash
git clone https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering.git
cd SVD-Based-Collaborative-Filtering
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python src/main.py --approve --approval-reason "reviewed validation results"   # ≈ 30 s
uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Docker: `docker compose --profile tools run --rm trainer` then `docker compose up -d api`.

## 9. Using the API

```bash
curl "http://127.0.0.1:8000/recommendations/1?n=5"
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' -d '{"user_id": 1, "movie_id": 50}'
./examples/demo/run_demo.sh           # walk-through of every endpoint
```

Swagger UI: http://127.0.0.1:8000/docs. Reference: [API_REFERENCE.md](API_REFERENCE.md).
Workflows: [USER_GUIDE.md](USER_GUIDE.md).

## 10. Tests and the MLOps workflow

```bash
ruff check src tests scripts
pytest -q                                                # 157 tests, offline, synthetic fixtures
python scripts/generate_synthetic_feedback.py --mode new_users --n 3000 --out fb.csv
python src/retrain.py --feedback fb.csv --dry-run        # gated retraining, changes nothing
python src/retrain.py --rollback --reason "..."          # re-approve the previous model
python src/run_movietweetings.py                         # separate MovieTweetings pipeline
```

Final audit results: [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md),
[FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md).

## 11. Important limitations

* **MovieLens licence: research use only.** Commercial or revenue-bearing use requires written
  permission (the University of Minnesota offers a negotiated commercial licence) or a
  commercially licensed / client-owned dataset. No permission has been obtained.
* **MovieTweetings**: MIT-licensed by its author, but derived from public tweets and linkable to
  Twitter accounts; its use needs the client's **legal review** ([docs/DATASET_LICENSING.md](docs/DATASET_LICENSING.md)).
* **Top-10 quality**: most-popular beats SVD offline (NDCG@10 0.147 vs 0.011 on MovieLens).
* **Retraining** was demonstrated with **synthetic** feedback only.
* **Automatic promotion is not production-safe as configured**: the gate accepts a candidate that
  merely ties the current model (`min_rmse_improvement` 0.0); configure a margin and a human
  approval step.
* The API has **no authentication, TLS or rate limiting**; provide them at a gateway
  ([SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md)).
* Random per-user split (not temporal); popularity-only cold start.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` 503 "no approved model" | run `python src/main.py --approve --approval-reason "..."` (or the Docker trainer) |
| Dataset download fails with a TLS error | the project ships the GroupLens intermediate certificate; behind a corporate proxy set `SSL_CERT_FILE` to your proxy CA, or copy a verified `ml-latest-small.zip` into `data/` |
| `SHA-256 mismatch` | the archive is not the expected MovieLens release; delete it and download again |
| Warning "No monitoring salt configured" | set `SVD_MONITORING_SALT` to a secret value for stable pseudonyms |
| API not reachable from another host | Compose binds 127.0.0.1 by design; expose only behind a TLS proxy (`API_BIND`) |
| `n must be <= 100` | `api.max_n` in `config.yaml` |

## 13. Documentation index

| Document | Purpose |
|---|---|
| [CLIENT_HANDOVER.md](CLIENT_HANDOVER.md) | what was delivered, release, next steps |
| [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md) | exact setup, Docker, MLflow, shutdown |
| [USER_GUIDE.md](USER_GUIDE.md) | workflows, examples, model management, monitoring, retraining |
| [API_REFERENCE.md](API_REFERENCE.md) | every endpoint with schemas and real responses |
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | architecture and implementation |
| [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md) | tests, CI, evaluation, final audit |
| [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md) | security, privacy, licensing, production prerequisites |
| [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md) | software and dataset licences |
| [FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md) | final QA checks with status |
| [CHANGELOG.md](CHANGELOG.md), [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md) | history, model and data facts |
| [docs/DATASET_LICENSING.md](docs/DATASET_LICENSING.md), [docs/MOVIETWEETINGS_EVALUATION.md](docs/MOVIETWEETINGS_EVALUATION.md) | dataset options and the alternative pipeline |
| [README.md](README.md) | full research and engineering documentation |
