# Client handover: SVD Collaborative Filtering Movie Recommender

## 1. What has been delivered

A movie recommender based on truncated SVD with a corrected, leakage-free evaluation (validation-only
model selection, test evaluated once, baselines and ranking metrics) and a complete MLOps layer:
FastAPI service with popularity fallback, model registry with explicit approval and rollback, MLflow
tracking, pseudonymised monitoring with drift detection, gated retraining from feedback, Docker images
and Compose, GitHub Actions CI with a real-data pipeline job, tests, a demo, a separate MovieTweetings
pipeline, a dataset-licensing assessment and full documentation.

| | |
|---|---|
| Release | **v1.0.0-mlops** |
| Repository | https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering |
| GitHub release | https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/releases/tag/v1.0.0-mlops |
| Release commit | `71b503d9f45c1eb3ace8aed50e76242731bfa67f` |
| Delivery package | release + post-release delivery commits on branch `client-delivery` (PR #4): client documentation, demo, OpenAPI export, MLflow-alias rollback fix; the exact commit is in `PACKAGE_MANIFEST.md` |
| Model | `svd-k20-fce385dc-c6b9d577` (reproduced in ≈ 30 s from the public dataset) |

## 2. Main capabilities

* `GET/POST /recommendations` (SVD for known users, popular movies for unknown users, exclusions),
  `POST /predict` (rating estimate with fallbacks), `POST /feedback`.
* `GET /monitoring/summary`, `GET /metrics`: drift, cold-start rate, observed RMSE, Prometheus.
* `python src/main.py`: download, validate, split, select k on validation, evaluate once, register.
* `python src/retrain.py`: gated retraining, dry run, rollback (MLflow alias follows).
* `python src/run_movietweetings.py`: the same methodology on MovieTweetings (separate registry).
* Docker: hardened API, trainer, MLflow.

## 3. How to install and run

[CLIENT_README.md](CLIENT_README.md) §8 (short) and [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md)
(exact, tested): Python 3.12 venv → `pip install -r requirements.txt -r requirements-dev.txt` →
`python src/main.py --approve --approval-reason "..."` → `uvicorn api:app --app-dir src` →
`./examples/demo/run_demo.sh`. Docker: `docker compose --profile tools run --rm trainer` then
`docker compose up -d api`. The MovieLens data is downloaded at first run (not redistributed).

## 4. Documentation index

| Start | [CLIENT_README.md](CLIENT_README.md) |
|---|---|
| Setup | [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md) |
| Daily use | [USER_GUIDE.md](USER_GUIDE.md), [API_REFERENCE.md](API_REFERENCE.md), `docs/openapi.json` |
| Internals | [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), ARCHITECTURE.md, MONITORING.md, RETRAINING.md, DEPLOYMENT.md |
| Quality | [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md), [FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md), FINAL_E2E_ACCEPTANCE.md |
| Risk and legal | [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md), SECURITY.md, [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md), docs/DATASET_LICENSING.md |
| Model and data | MODEL_CARD.md, DATA_CARD.md, docs/MOVIETWEETINGS_EVALUATION.md, outputs/HISTORICAL_RESULTS.md |
| History | [CHANGELOG.md](CHANGELOG.md), `docs/releases/v1.0.0-mlops.md` |

## 5. Testing evidence

Final delivery QA from a fresh clone: 157 tests passed (83 % coverage), lint clean, dependency audit
clean after the documented pip upgrade, the real-data pipeline reproduced every published metric,
retraining rejected noise and promoted genuine new-user feedback, rollback restored both the registry
and the MLflow alias, API and Docker checks passed (including hardening and persistence), MLflow
server and container started, the MovieTweetings pipeline reproduced its results, and GitHub
Actions is green on the release and delivery commits. Details: FINAL_DELIVERY_QA_REPORT.md,
`docs/evidence/`.

## 6. Known limitations

* **MovieLens is licensed for research use only**; commercial use needs written permission or a
  commercially licensed / client-owned dataset. **MovieTweetings** needs legal review.
* Offline top-10 quality: most popular beats SVD (NDCG@10 0.147 vs 0.011); SVD's strength is
  rating prediction (RMSE 0.8853 vs 0.8943).
* Retraining demonstrated with synthetic feedback only; the default gate promotes a tie
  automatically, which is not production-safe without a margin and human approval.
* No authentication, TLS or rate limiting in the application; `/feedback` stores raw user ids.

## 7. Client-side prerequisites

* Python 3.12+ or Docker 24+; internet access to files.grouplens.org on the first run.
* For commercial use: a dataset licence decision (docs/DATASET_LICENSING.md) and a gateway with TLS,
  authentication, authorisation and rate limiting.

## 8. Recommended next steps

1. Install and run the demo; review FINAL_DELIVERY_QA_REPORT.md.
2. Decide the data route: a MovieLens commercial licence, legal clearance of MovieTweetings, or
   (recommended) your own first-party ratings.
3. Run an online comparison of SVD vs most-popular recommendations with real users.
4. Configure the promotion policy (`min_rmse_improvement` > 0, human approval) and retention for
   feedback and monitoring logs.
5. Deploy behind a TLS gateway with authentication and rate limiting; integrate `/metrics`.
6. Agree licence/ownership terms for the code (LICENSE_AND_DATA_ATTRIBUTION.md §1).
