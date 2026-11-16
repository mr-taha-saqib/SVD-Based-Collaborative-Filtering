# Delivery summary: SVD movie recommender

**Status: delivered on branch `mlops-production-delivery`; acceptance PASS except one
CI job that could not be triggered (see FINAL_E2E_ACCEPTANCE.md).**

| Area | Delivered | Verified |
|---|---|---|
| Methodology fix | test-set rank selection replaced by train/validation/test; historical results labelled | yes, leakage tests + real run |
| Model | truncated SVD, k = 20 chosen on validation; test RMSE 0.8853 | yes |
| API | FastAPI, 9 endpoints, Swagger at `/docs` | yes, 24 API tests + curl transcript |
| Docker | API (396 MB) + trainer images, compose with persistent volumes | yes, container inference + restart |
| MLflow | local SQLite tracking, model versions with `champion` alias | yes, run `4e316c2b…` |
| Monitoring | Prometheus metrics, PSI drift, delayed-label performance | yes, controlled drift test |
| Retraining | gated promotion, dry-run, rollback, reports | yes, promote + reject + rollback executed |
| CI | GitHub Actions: lint, tests 3.12/3.13, Docker smoke test | yes, run 43967996915 green |
| Tests | 143 (was 48; 8 of the original failed with pandas 3) | yes |

**Two things the client must know**

1. MovieLens is licensed for research only; commercial use needs GroupLens permission.
2. Offline, the SVD model's top-10 recommendations score far below a most-popular list
   (NDCG@10 0.011 vs 0.147); its strength is rating prediction. Validate rankings online.

Start: `README.md` → `DEPLOYMENT.md` → `API_DOCUMENTATION.md`.
