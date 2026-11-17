# Delivery summary: SVD movie recommender

**Status: delivered on branch `mlops-production-delivery` (PR #1, open, not merged);
acceptance PASS, including the real-data CI job (see FINAL_E2E_ACCEPTANCE.md). Not
commercially production-ready (licence, ranking quality, no auth/TLS).**

| Area | Delivered | Verified |
|---|---|---|
| Methodology fix | test-set rank selection replaced by train/validation/test; historical results labelled | yes, leakage tests + real run |
| Model | truncated SVD, k = 20 chosen on validation; test RMSE 0.8853 | yes |
| API | FastAPI, 9 endpoints, Swagger at `/docs` | yes, 24 API tests + curl transcript |
| Docker | API (396 MB) + trainer images, compose with persistent volumes | yes, container inference + restart |
| MLflow | local SQLite tracking, model versions with `champion` alias | yes, run `4e316c2b…` |
| Monitoring | Prometheus metrics, PSI drift, delayed-label performance | yes, controlled drift test |
| Retraining | gated promotion, dry-run, rollback, reports | yes, promote + reject + rollback executed |
| CI | GitHub Actions: lint, tests 3.12/3.13, Docker smoke test, real-data pipeline | yes, runs 44072171688 (push) and 44072182735 (PR) green on `75976e4` |
| Release | notes in `docs/releases/v1.0.0-mlops.md` | tag/release **not published**: the delivery session may push branches only (see FINAL_E2E_ACCEPTANCE.md §7) |
| Tests | 145 (was 48; 8 of the original failed with pandas 3) | yes |

**Two things the client must know**

1. MovieLens is licensed for research only. Commercial use needs written permission from
   GroupLens or a commercially licensed / client-owned replacement dataset.
2. Offline, the SVD model's top-10 recommendations score far below a most-popular list
   (NDCG@10 0.011 vs 0.147); its strength is rating prediction. Validate rankings online.

Start: `README.md` → `DEPLOYMENT.md` → `API_DOCUMENTATION.md`.
