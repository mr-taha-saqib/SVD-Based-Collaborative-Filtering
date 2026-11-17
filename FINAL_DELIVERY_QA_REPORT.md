# Final delivery QA report: SVD Collaborative Filtering Movie Recommender

| | |
|---|---|
| Release | `v1.0.0-mlops` → commit `71b503d9f45c1eb3ace8aed50e76242731bfa67f` (annotated tag, GitHub release published) |
| QA'd commit | `15c60d19a8eee53dd869f6ee1bdab4ea897355f4` (branch `client-delivery`, PR #4): the release plus a post-release delivery commit (documentation, demo, OpenAPI export, MLflow-alias rollback fix + test). The QA report, testing and handover documents and later documentation corrections were added afterwards in documentation-only commits |
| Date | 2026-11-17/18 (UTC) |
| Environment | Ubuntu 24.04.5, x86-64, Python 3.12.3, Docker (BuildKit), CPU only; outbound traffic via a TLS-inspecting proxy |
| Method | fresh `git clone`, new virtual environment, documented commands; full log: [`docs/evidence/16_final_delivery_qa.log`](docs/evidence/16_final_delivery_qa.log) |

Statuses: **PASS**, **FAIL**, **BLOCKED**, **NOT RUN**, **NOT APPLICABLE**. No check failed.

## 1. Installation and static checks

| # | Check | Command (abridged) | Result | Status |
|---|---|---|---|---|
| 1 | Source baseline | `git describe`, `git diff --stat v1.0.0-mlops HEAD` | tag reachable; 19 files changed vs the release (docs, demo, `src/retrain.py`, `src/tracking.py`, `src/run_movietweetings.py`, one test) | PASS |
| 2 | Clean environment | `python3.12 -m venv .venv` | Python 3.12.3 | PASS |
| 3 | Dependency installation | `pip install -r requirements.txt -r requirements-dev.txt` | 100 distributions | PASS |
| 4 | Dependency consistency | `pip check` | no broken requirements | PASS |
| 5 | Dependency vulnerabilities | `pip-audit --path .venv/…/site-packages` | project dependencies: none. The venv's bundled **pip 24.0** (from Ubuntu's Python) had 12 advisories; after `python -m pip install --upgrade pip` (pip 26.3): no known vulnerabilities. The upgrade step was added to INSTALLATION_AND_SETUP.md | PASS (after documented step) |
| 6 | Lint | `ruff check src tests scripts` | all checks passed | PASS |
| 7 | Formatting | – | no formatter configured | NOT APPLICABLE |
| 8 | Compose configuration | `docker compose config -q` | valid; services `api`, `trainer`, `mlflow` | PASS |

## 2. Automated tests

| # | Check | Command | Result | Status |
|---|---|---|---|---|
| 9 | Unit, API, data, ranking, registry, monitoring, retraining, MovieTweetings tests | `pytest --ignore=tests/test_integration.py` | 152 passed | PASS |
| 10 | Integration tests | `pytest tests/test_integration.py` | 5 passed (pipeline + selection never sees test, MLflow run + alias, alias after rollback, full lifecycle) | PASS |
| 11 | Full suite + coverage | `pytest --cov=src` | **157 passed**, coverage 83 % of `src/` | PASS |

## 3. End-to-end, API and MLOps workflow

| # | Check | Result | Status |
|---|---|---|---|
| 12 | Real-data pipeline | `python src/main.py --approve …`: fresh MovieLens download, SHA-256 verified; k = 20 selected on validation (RMSE 0.8729); test RMSE 0.8853, MAE 0.6754; version `svd-k20-fce385dc-c6b9d577` (= delivered version) | PASS |
| 13 | Results vs committed | `outputs/corrected/final_metrics.json`: only run metadata differs (timestamp, commit, MLflow run id, platform, duration); all metrics identical | PASS |
| 14 | Retraining: rejection | 3,000 random-noise synthetic ratings → `rejected_by_gate`, approved unchanged | PASS |
| 15 | Retraining: promotion | 3,000 new-user synthetic ratings → `promoted` (`svd-k20-fce385dc-9180e770`); MLflow `champion` → version 3 | PASS |
| 16 | Rollback + MLflow alias | `python src/retrain.py --rollback`: registry back to `c6b9d577`; MLflow `champion` back to version 1 (the fix delivered in this commit) | PASS |
| 17 | Local API + health | `uvicorn api:app --app-dir src` → `/health` 200 | PASS |
| 18 | API example requests | `examples/demo/run_demo.sh`: all 10 steps (SVD list, popularity fallback, exclusions, prediction, 422, feedback 202, summary, metrics) | PASS |
| 19 | Monitoring, normal traffic | 200 synthetic requests: input drift `ok` (PSI 0.0283), prediction drift `ok`, cold start 0.5 % | PASS |
| 20 | Drift detection | fresh process, unknown-user traffic: input drift `alert` (PSI 16.78), cold start 100 %; random-salt warning logged when the salt is unset | PASS |
| 21 | MLflow server startup + connectivity | `mlflow server` on the run's DB: `/health` OK; experiment `svd-movie-recommender`; 3 runs; `champion` → version 1 | PASS |
| 22 | MovieTweetings pipeline | `python src/run_movietweetings.py`: checksum-pinned download (`users.dat` not downloaded); 921,398 → 593,204 ratings; k = 5; test RMSE 0.6930; same version id `svd-k5-cbc6d187-daaead1d`; metrics identical to the committed results | PASS |
| 23 | MovieLens 1M cross-check | not repeated (executed earlier, `docs/evidence/08_ml1m_run.log`) | NOT RUN (optional) |

## 4. Docker

| # | Check | Result | Status |
|---|---|---|---|
| 24 | Build API image | `svd-recommender-api:1.0.0` `4b4315fccb7a`, 423 MB (with the documented `--secret id=pip_ca` for the proxy) | PASS |
| 25 | Build trainer image | `svd-recommender-trainer:1.0.0` `9da031b7890f`, 1.31 GB | PASS |
| 26 | Train in the container | trainer on the Compose volumes: k = 20, test RMSE 0.8853, approved `svd-k20-fce385dc-6cfca4f0` (the id contains an artifact hash; metrics identical). Run with host networking because the sandbox proxy listens on localhost (environment-specific) | PASS |
| 27 | Start + health | `docker compose up -d api` → healthy; `/health` 200 | PASS |
| 28 | Demo script against the container | all 10 steps | PASS |
| 29 | Invalid inputs | user 0, `n=500`, rating 6 → 422 | PASS |
| 30 | Hardening | published on `127.0.0.1:8000`; user `app` (uid 10001); `CapDrop [ALL]`; `no-new-privileges`; `models` mounted read-only (write refused); code not writable | PASS |
| 31 | Persistence | healthy after restart; feedback file and monitoring events kept | PASS |
| 32 | MLflow container (`--profile mlflow`) | `ghcr.io/mlflow/mlflow:v3.18.0` started; `/health` OK; trainer's experiment visible | PASS |
| 33 | Clean-up | `docker compose --profile tools --profile mlflow down -v`: 0 containers, 0 volumes left | PASS |

## 5. Documentation and CI

| # | Check | Result | Status |
|---|---|---|---|
| 34 | Relative documentation links | 38 Markdown files, 63 relative links (files and anchors); only broken targets at QA time were the three documents written after QA; re-checked at packaging | PASS (re-checked at packaging) |
| 35 | External links | GitHub PR pages HTTP 200; Actions runs verified via the GitHub API | PASS |
| 36 | Documentation vs behaviour | API_REFERENCE examples captured from the service; [Tested] commands executed in this QA; corrections made: health-check interval, image size, pip upgrade step | PASS |
| 37 | GitHub Actions on the QA'd commit | run [44153466879](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/actions/runs/44153466879): lint + tests (3.12, 3.13), Docker smoke test, real-data pipeline: all success | PASS |
| 38 | GitHub Actions on the release commit | run 44088400666 on `71b503d`: success | PASS |
| 39 | ZIP integrity | performed after packaging; results in the delivery folder copy of this report and in `PACKAGE_MANIFEST.md` | see package verification |

## 6. Findings fixed during this QA

* MLflow `champion` alias did not follow a rollback (fixed + tested; check 16).
* Fresh Ubuntu venvs carry pip 24.0 with advisories (installation step added; check 5).
* Documentation details corrected (health-check interval, image size).

## 7. Conclusion

All executable checks pass. Outstanding items are external: a dataset licence decision (MovieLens
research-only; MovieTweetings legal review), production controls (TLS, authentication, rate
limiting), a stricter promotion policy, and online validation of recommendation quality. The
package is **delivery-ready**; it is **not production-ready**.
