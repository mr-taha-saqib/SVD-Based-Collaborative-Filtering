# Testing and validation

Release `v1.0.0-mlops`. Commands, suites, CI, evaluation and the actual results of the final
delivery audit ([FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md)). Suite details:
[TESTING.md](TESTING.md).

## 1. Environment requirements

Python 3.12 or 3.13; `pip install -r requirements.txt -r requirements-dev.txt`. Tests use synthetic
fixtures and need no network or dataset. The real-data pipeline needs access to files.grouplens.org.

## 2. Commands

```bash
ruff check src tests scripts                    # lint
pytest -q                                       # 157 tests, ≈ 40 s
pytest -q --cov=src                             # with coverage
pytest -q tests/test_integration.py             # integration only
python src/main.py --approve --approval-reason "..."      # end-to-end on real data (≈ 30 s)
python src/run_movietweetings.py                # alternative dataset pipeline
./examples/demo/run_demo.sh                     # API walk-through (running API needed)
```

`src/main.py` rewrites `outputs/corrected/`; restore the committed files with
`git checkout -- outputs/` (only run metadata changes).

## 3. Test suites

| File | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 49 | original suite (strengthened) + historical-label and validation-argmin checks |
| `test_data_layer.py` | 20 | validation rules, duplicates, fingerprint, 3-way split (disjoint, complete, deterministic), leakage detector, checksum-verified extraction, path traversal, config |
| `test_ranking.py` | 5 | top-N masking, NDCG, oracle scorer, recall denominator |
| `test_recommender.py` | 10 | artifact reload, exclusions, cold start, titles, fallbacks, tamper detection |
| `test_registry.py` | 11 | register ≠ approve, reasons, promotion, rollback, rejection, idempotence, unsafe ids, readable version directories |
| `test_monitoring.py` | 9 | PSI, statuses, normal vs drift, delayed-label performance, pseudonymised log, salt handling |
| `test_retrain.py` | 17 | every gate criterion, feedback validation, **test partition never read**, dry run, reports, rollback errors |
| `test_api.py` | 24 | all endpoints, schemas, 422 cases, cold start, exclusions, 503 without model, reload auth |
| `test_movietweetings.py` | 7 | preparation, 20-core, duplicates, `users.dat` never opened, checksum mismatch, separate locations |
| `test_integration.py` | 5 | pipeline outputs, **selection never sees test**, MLflow run + `champion` alias, **alias after rollback**, full lifecycle (pipeline → API → retraining) |

Coverage (final QA): **83 %** of `src/` lines.

## 4. Continuous integration

`.github/workflows/ci.yml`: lint + tests on Python 3.12 and 3.13; Docker build + container smoke
test; full MovieLens pipeline with assertions (zero train/test overlap, validation-based selection,
dataset checksum). Final runs: delivery commit `15c60d1` → run 44153466879 (4/4 jobs success);
release commit `71b503d` → run 44088400666 (success).

## 5. Docker validation (final QA)

API (423 MB) and trainer (1.31 GB) images built; containerised training reproduced k = 20 and test
RMSE 0.8853; API healthy; demo and invalid-input checks passed; hardening verified (localhost port,
uid 10001, capabilities dropped, read-only model volume, code not writable); state persisted across a
restart; MLflow container started.

## 6. Model evaluation (MovieLens latest-small, test evaluated once)

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|---|---|
| Global mean | 1.0442 | 0.8266 | – | – | – | – |
| User mean | 0.9435 | 0.7350 | – | – | – | – |
| Item mean | 0.9782 | 0.7544 | – | – | – | – |
| User + item bias | 0.8943 | 0.6834 | 0.0028 | 0.0009 | 0.0024 | 0.0054 |
| **SVD k = 20** | **0.8853** | **0.6754** | 0.0103 | 0.0022 | 0.0109 | 0.0294 |
| SGD-MF k = 20 | 0.8737 | 0.6708 | – | – | – | – |
| **Most popular** | – | – | **0.1108** | **0.0923** | **0.1470** | 0.0186 |

**Rating prediction vs top-N:** SVD predicts ratings slightly better than the bias baseline, but for
top-10 recommendation the most-popular list is far better offline. SGD-MF (a comparison model, not
served) has a lower RMSE than SVD. MovieTweetings results: docs/MOVIETWEETINGS_EVALUATION.md.

## 7. Actual results of the final audit

| Area | Result |
|---|---|
| Install, `pip check`, lint | PASS |
| pytest | **157 passed** (152 + 5 integration), coverage 83 % |
| Dependency audit | PASS after the documented pip upgrade (bundled pip 24.0 had advisories) |
| Real-data pipeline | PASS: published metrics reproduced exactly |
| Retraining reject / promote / rollback, MLflow alias | PASS |
| API local + Docker, demo, invalid inputs, hardening, persistence | PASS |
| Monitoring normal / drift | PASS (ok / alert) |
| MLflow server and container | PASS |
| MovieTweetings pipeline | PASS: metrics reproduced |
| MovieLens 1M | NOT RUN in the final audit (executed earlier) |

## 8. Known test limitations

* Retraining is exercised with **synthetic** feedback only.
* Ranking quality is measured offline; no online/A-B evaluation.
* No load, performance or penetration testing.
* Random (not temporal) split; results may be optimistic relative to forward-in-time use.
