# Changelog

## 1.0.0 additions (2026-11-17, session 3, PR #3)

### Security (see SECURITY.md)
* Constant-time admin-token comparison for `/model/reload`.
* Unset or placeholder `SVD_MONITORING_SALT` no longer produces unsalted (reversible)
  pseudonyms: a random per-process salt is used instead.
* Model registry version directories are created 0755 instead of 0700, so a registry can be
  served read-only by another user (the uid-10001 container).
* API image: code and config are no longer writable by the runtime user; pip 26.3 /
  setuptools 84.2.1 replace the base image's versions with published advisories.
* Compose: API and MLflow published on 127.0.0.1 by default (`API_BIND`), `cap_drop: ALL`,
  `no-new-privileges`, read-only `models` volume for the API.

### Added
* Separate MovieTweetings pipeline (`src/movietweetings.py`, `src/run_movietweetings.py`),
  results in `outputs/external/movietweetings/`, `docs/MOVIETWEETINGS_EVALUATION.md`.
* `docs/DATASET_LICENSING.md` (MovieLens commercial route, alternatives, MovieTweetings terms).
* 11 tests (156 total).

### Unchanged
* MovieLens methodology, data, split, selected k and all published results.

## 1.0.0 (2026-11-17), branch `mlops-production-delivery`

### Fixed
* **Test-set leakage in model selection**: `src/main.py` and `src/run_ml1m.py` chose
  the SVD rank by minimum *test* RMSE and reported that same test score. Replaced by a
  train/validation/test split with selection on validation and one final test evaluation.
* `train_test_split` crashed under pandas ≥ 3 (`ValueError: array is read-only`); 8 of
  48 tests failed with current pandas. Fixed by copying the index array.
* Dataset download failed with a TLS error because GroupLens omits its intermediate
  certificate; the public intermediate is now shipped and the archive checksum verified.
* README claimed rank selection "by cross-validation" (none was performed); corrected.
* Unpinned dependencies and a missing `PyYAML` requirement.
* Weak tests strengthened (see `tests/test_pipeline.py` header): NaN semantics, bias
  centring, seed comparison, and a "recomputation" test that only range-checked a
  historical file.

### Added
* Data validation, dataset fingerprints, split manifest with leakage checks.
* Ranking metrics (Precision/Recall/NDCG/HitRate@K, catalogue coverage) and a
  most-popular baseline.
* Versioned model registry with explicit approval, rejection records and rollback.
* FastAPI service, structured JSON logging, Prometheus metrics, drift monitoring,
  feedback endpoint.
* MLflow tracking (local SQLite) with model registry aliases.
* Gated retraining workflow with dry-run, idempotence and reports.
* Dockerfile (API + trainer targets), docker-compose with named volumes, `.env.example`.
* GitHub Actions CI; 145 tests (was 48).
* Documentation set: ARCHITECTURE, API_DOCUMENTATION, MODEL_CARD, DATA_CARD,
  DEPLOYMENT, MONITORING, RETRAINING, TESTING, CONTRIBUTING, FINAL_E2E_ACCEPTANCE.

### Preserved
* Historical results and plots in `outputs/` (labelled in `outputs/HISTORICAL_RESULTS.md`),
  LaTeX report and slides.

## 0.x (before 2026-11)
Academic study: SVD on MovieLens latest-small and 1M, baselines, SGD-MF, figures.
