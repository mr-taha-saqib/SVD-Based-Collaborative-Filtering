# Testing

```bash
pip install -r requirements-dev.txt
pytest                                   # 156 tests, ~35 s, offline, synthetic fixtures
pytest -m "not performance"              # what CI runs (skips the timing-based check)
pytest --cov=src --cov-report=term       # coverage (83 % of src/ in the acceptance run)
ruff check src tests scripts
```

No test downloads data, touches `models/`, `outputs/`, `mlflow.db` or `monitoring/`:
fixtures redirect every path into pytest's temporary directory (`tests/conftest.py`).
Synthetic ratings (`scripts/synthetic_data.py`) have a planted low-rank structure so
that "SVD beats the global mean" is a meaningful assertion.

## Inventory

| File | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 49 | original suite (48, five assertions strengthened) + historical-label and validation-argmin checks |
| `test_data_layer.py` | 20 | validation rules, duplicates, fingerprint, 3-way split (disjoint, complete, deterministic, order-invariant), leakage detector, checksum-verified extraction, path traversal, config |
| `test_ranking.py` | 5 | top-N masking, NDCG values, oracle scorer, recall denominator |
| `test_recommender.py` | 10 | artifact reload equality, exclusion of rated items, cold start, titles, fallback methods, tamper detection |
| `test_registry.py` | 11 | register ≠ approve, reasons required, promotion keeps previous, rollback, rejection, idempotence, unsafe ids, version directories readable by other users |
| `test_monitoring.py` | 9 | PSI, statuses, normal vs controlled drift, delayed-label performance, pseudonymised log, random salt when unset or placeholder, configured salt used |
| `test_retrain.py` | 17 | every gate criterion, feedback validation, data requirements, promote/reject, dry run, **test partition never read**, mismatched base data, unique report names, clean rollback error |
| `test_api.py` | 24 | all endpoints, schemas, 422 cases, malformed JSON, cold start, exclusions, 503 without model, reload auth (missing / wrong / near-miss token), OpenAPI |
| `test_movietweetings.py` | 7 | rescaling and zero-rating drop, 20-core, duplicate handling, out-of-range ratings, movie parsing, `users.dat` never opened, checksum mismatch aborts, separate output/registry/MLflow locations |
| `test_integration.py` | 4 | pipeline outputs + validation argmin, **selection never sees test**, MLflow run + `champion` alias, full lifecycle (pipeline → API → retraining) |

## Review of the original suite

Baseline (commit `cae97e2`): 48 tests; 48 pass with pandas 2.3, but **8 fail with
pandas 3.0** (`ValueError: array is read-only` in `train_test_split`), the version an
unpinned `pip install -r requirements.txt` resolves today. Fixed in `data_loader.py`.

Assertions that could pass without testing their stated behaviour were strengthened,
not removed: matrix NaN semantics, bias-centring (now checks the exact zero column
means), seed-difference comparison, and `test_saved_csv_rmse_matches_recomputed`
(which only range-checked a historical file) replaced by checks that the historical
results are preserved and labelled and that the corrected `k` is the argmin of the
validation sweep. The timing-based batch-vs-loop test is marked `performance`.

## Beyond pytest (executed, evidence in `docs/evidence/`)

* Real-data pipeline: `01_pipeline_run_real_data.log` (local) and
  `09_github_real_data_job_push.log` / `09_github_real_data_job_pull_request.log` (GitHub Actions)
* Session-2 acceptance audit: `10_session2_acceptance_audit.log`,
  `11_session2_retraining_rollback_audit.log`
* API curl session: `02_api_curl_examples.txt`
* Docker trainer + API container session: `03_docker_trainer_run.log`, `04_docker_api_session.log`
* Monitoring + controlled drift: `05_monitoring_drift_test.log`
* Retraining scenarios: `06_retraining_e2e.log`, `retraining_reports/`
* Test run: `07_pytest_output.txt`, `07_pytest_junit.xml`
* Session 3: clean-install run `15_session3_clean_install_tests.log` (156 passed), hardened Docker
  `12_session3_docker_hardened_compose.log`, monitoring + retraining `13_session3_monitoring_retraining_movielens.log`,
  MLflow `14_session3_mlflow_records.log`, MovieTweetings `movietweetings/01-04_*.log`
