# Final end-to-end acceptance report: SVD movie recommender

| | |
|---|---|
| Repository | https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering |
| Branch | `mlops-production-delivery` (from `main` @ `cae97e2`) |
| Environment | Linux x86_64, 2 CPU cores, 8 GB RAM, Python 3.12.3 (3.13.18 also tested), Docker 29.9.3 |
| Date | 2026-11-17 |
| Approved model | `svd-k20-fce385dc-c6b9d577` (k = 20, MLflow run `4e316c2b14884868b640f5f46af69b96`) |

Every row below was executed and its output observed. Evidence files are in
`docs/evidence/`. Nothing is marked PASS without execution.

## 1. Baseline before changes

| Item | Observed |
|---|---|
| Commit | `cae97e2` on `main`, 25 files |
| Tests | 48 collected. **pandas 3.0.5: 8 failed / 40 passed** (`ValueError: array is read-only` in `train_test_split`). pandas 2.3.3: 48 passed |
| Critical defect | `src/main.py` and `src/run_ml1m.py` selected rank `k` by minimum **test** RMSE and reported that test RMSE |
| Other defects | README claimed "cross-validation" (none performed); dataset download failed TLS verification (GroupLens omits its intermediate certificate); `PyYAML` missing from requirements; unpinned dependencies; 5 weak test assertions; no API, Docker, MLflow, monitoring, retraining or CI |
| Artifacts | historical results/plots in `outputs/` (preserved, labelled) |

## 2. Acceptance table

| Requirement | Test executed | Result | Evidence | Status |
|---|---|---|---|---|
| Dataset download + validation | `python src/main.py` (fresh `data/`) | archive SHA-256 verified (MD5 equals GroupLens' published value); 100,836 ratings, 0 invalid, 0 duplicates | `outputs/corrected/data_quality.json`, `01_pipeline_run_real_data.log` | PASS |
| Preprocessing | same run | bias centring + zero imputation on train matrix 610 × 8,578 | log | PASS |
| Train / validation / test split | same run | 70,579 / 10,093 / 20,164; split SHA-256 `c5574af8…` | `outputs/corrected/split_manifest.json` | PASS |
| Leakage verification | same run + `test_selection_never_sees_test`, `test_partitions_disjoint_and_complete` | 0 shared (user, movie) pairs in all three partition pairs; selection receives validation only | split manifest, pytest | PASS |
| SVD training | same run | full SVD 0.6 s | log | PASS |
| Validation-based rank selection | same run | k = 20 (validation RMSE 0.8729; sweep 0.8755 … 0.8820) | `k_sweep_validation.csv` | PASS |
| Baseline comparison | same run | global/user/item mean, user+item bias, SGD-MF, most popular (validation and test) | `results_validation.csv`, `results_test.csv` | PASS |
| Final test evaluation | same run (once, after selection) | SVD test RMSE 0.8853, MAE 0.6754, NDCG@10 0.0109; most popular NDCG@10 0.1470 | `results_test.csv`, `final_metrics.json` | PASS |
| ML-1M cross-check | `python src/run_ml1m.py` | 1,000,209 ratings; k = 20 selected on validation (RMSE 0.9013); test RMSE 0.9013, MAE 0.7070 vs user+item bias 0.9318; randomised SVD; 15 s | `outputs/corrected/ml1m/` | PASS |
| Artifact saving | same run | `models/versions/svd-k20-fce385dc-c6b9d577/` (npz + metadata + movies + reference) | registry | PASS |
| Artifact reloading | API startup; `test_reload_reproduces_predictions` | identical predictions after reload; tampered file rejected | pytest, `02_api_curl_examples.txt` | PASS |
| Reproducibility | pipeline re-run | identical version id `svd-k20-fce385dc-c6b9d577` (same content hash); Docker trainer: same metrics to 4 dp, different content hash (BLAS float differences) | logs | PASS |
| MLflow logging | same run | run `4e316c2b14884868b640f5f46af69b96`: params, sweep metrics per k, val/test metrics, fingerprints, artifacts; model version 1 alias `champion` | `mlflow.db`, log | PASS |
| Recommendation generation / top-N | `GET /recommendations/1?n=3` | 3 ranked items with titles, genres, scores | `02_api_curl_examples.txt` | PASS |
| Already-rated exclusion | `test_recommendations_exclude_rated_and_are_sorted`, API test | no rated movie returned | pytest | PASS |
| Unknown user / cold start | `GET /recommendations/999999?n=2` | `popularity_fallback`, `cold_start: true` (Forrest Gump, Shawshank) | curl transcript | PASS |
| Unknown movie ids | `POST /recommendations` with `99999999`; `POST /predict` | reported in `unknown_excluded_movie_ids`; predict uses `user_bias_only` | curl transcript | PASS |
| FastAPI startup | `uvicorn api:app --app-dir src` | healthy, model loaded | curl transcript | PASS |
| Valid API request | curl | 200 with schema | curl transcript | PASS |
| Invalid API requests | user 0, n = 500, missing field, rating 7, malformed JSON | 422 with structured error | curl transcript, 24 API tests | PASS |
| Model-not-available | `test_model_unavailable_returns_503` | `/health` 503, endpoints 503 | pytest | PASS |
| Monitoring execution | live API + `simulate_traffic.py` | rates, latency, PSI, Prometheus metrics | `05_monitoring_drift_test.log` | PASS |
| Controlled drift test | 300 normal vs 300 unknown-user requests | input PSI 0.0078 ok → 16.78 **alert**; cold start 0 % → 100 %; performance only after 40 labels | `05_monitoring_drift_test.log` | PASS |
| Retraining trigger | `src/retrain.py` with synthetic feedback (registry copy) | 6 scenarios executed | `06_retraining_e2e.log` | PASS |
| Candidate evaluation | same | candidate and approved compared on validation (base + feedback-validation) | `retraining_reports/*.md` | PASS |
| Model promotion / rejection | same | noise feedback → **rejected** (`…baf57cab`, run `6a7df9cf…`); new-user feedback → **promoted** (`…9180e770`, RMSE 0.8671 → 0.8604, run `a56a3717…`) | reports, MLflow model versions 1–2 | PASS |
| Previous-model preservation | same | previous version kept; rollback restored `svd-k20-fce385dc-c6b9d577` | `06_retraining_e2e.log` | PASS |
| Automated tests | `pytest --cov=src` | **143 passed**, 83 % coverage | `07_pytest_output.txt`, `07_pytest_junit.xml` | PASS |
| Lint | `ruff check src tests scripts` | all checks passed | – | PASS |
| Docker image build | `docker build --target base` / `trainer` | `svd-recommender-api:1.0.0` 396 MB; trainer 1.28 GB | build logs | PASS |
| Docker trainer | `docker compose --profile tools run --rm trainer` | trained, evaluated, approved into volume (dataset seeded from verified archive; see note 2) | `03_docker_trainer_run.log` | PASS |
| Docker container startup + health | `docker compose up -d api` | `(healthy)`, runs as uid 10001 | `04_docker_api_session.log` | PASS |
| Inference inside Docker | curl to container | svd + popularity fallback responses, 422 on invalid | `04_docker_api_session.log` | PASS |
| Persistence across restart | `docker compose restart api` | same model; event log and feedback retained | `04_docker_api_session.log` | PASS |
| Logs + clean stop | `docker compose logs`, `down` | JSON logs; containers removed, volumes kept | `04_docker_api_session.log` | PASS |
| GitHub Actions | push to `mlops-production-delivery` | run [43967692965](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/actions/runs/43967692965): **failed** on Python 3.11 (numpy 2.5 needs ≥ 3.12). Fixed; run [43967996915](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/actions/runs/43967996915): lint+tests 3.12 ✔, 3.13 ✔ (142 passed), Docker build + container inference ✔ | Actions | PASS |
| Real-data CI job | `workflow_dispatch` / push to `main` | not triggered: runs only on `main` or manual dispatch, and this session's GitHub token could not dispatch workflows | – | BLOCKED (executed locally instead) |
| Documentation | files present, commands executed | README + 11 documents; every command in README executed in this run | repo | PASS |
| Security review | `git grep` for keys/tokens/private keys; `.env` ignored | 0 findings; no secrets, datasets or models committed | – | PASS |

## 3. Failures found during acceptance and how they were resolved

1. **8 tests failing under pandas 3** (baseline). Fixed in `data_loader.train_test_split`; re-run: pass.
2. **Dataset download TLS failure**: missing intermediate + missing emSign root in the
   local trust store. Fixed by shipping the public intermediate and enabling partial-chain
   verification against it (Python 3.13's default behaviour). Docker in this sandbox:
   intercepted TLS (rejected, correctly), then HTTP 503 from the sandbox egress for
   GroupLens; the `dataset` volume was seeded with the archive downloaded and verified on
   the host, and the trainer re-verified its SHA-256.
3. **MLflow 3 `register_model` rejected plain artifact folders**; switched to
   `create_model_version` from the run artifact URI. Re-run: pass.
4. **Prediction-drift PSI was computed from popularity-fallback items** too; restricted
   to SVD responses. Re-run: pass.
5. **`approved_after` was null** on the data-requirement rejection path; fixed.
6. **CI failed on Python 3.11** (pinned numpy needs 3.12). Python floor raised to 3.12,
   matrix 3.12/3.13; re-run green.
7. Docker build behind the sandbox's TLS-inspecting proxy needed its CA: added an
   optional BuildKit secret (`pip_ca`), no CA in image layers.

## 4. Locations

* Model artifacts: `models/versions/<id>/` (git-ignored; reproducible with `python src/main.py --approve`)
* MLflow: `mlflow.db`, `mlartifacts/` (local); UI `mlflow ui --backend-store-uri sqlite:///mlflow.db`
* Results: `outputs/corrected/`; historical: `outputs/` + `outputs/HISTORICAL_RESULTS.md`
* Logs: `outputs/corrected/run.log`, `monitoring/events.jsonl`, `docs/evidence/`
* Docker images: `svd-recommender-api:1.0.0`, `svd-recommender-trainer:1.0.0`

## 5. Known limitations and external dependencies

* **MovieLens licence forbids commercial use without GroupLens permission.**
* Offline top-N quality of SVD is far below most-popular (NDCG@10 0.011 vs 0.147).
* Retraining was demonstrated with synthetic feedback only.
* No public deployment; no API authentication; monitoring state is per process.
* GroupLens availability is required for the first data download.

## 6. Decision

All mandatory acceptance items passed except the real-data CI job, which was
executed locally but could not be triggered on GitHub from this session. The project
is ready for client **handover as a reproducible, tested, containerised service and
reference implementation**. It is not ready for commercial production use until the
MovieLens licence question is resolved and the ranking-quality finding is addressed.
