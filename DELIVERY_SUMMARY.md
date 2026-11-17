# Delivery summary: SVD movie recommender

**Status: merged into `main` (PR #1, PR #2, PR #3); acceptance PASS, including the real-data
CI job (see FINAL_E2E_ACCEPTANCE.md). Not commercially production-ready (dataset licence,
ranking quality, deployment security prerequisites in SECURITY.md).**

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
| Tests | 156 (was 48; 8 of the original failed with pandas 3) | yes |
| Security review | `SECURITY.md`: 5 fixes (token comparison, pseudonymisation salt, image permissions, registry permissions, base-image pip/setuptools); localhost-only, read-only, capability-dropped Compose | yes, tests + container session |
| Alternative dataset | MovieTweetings pipeline (separate registry/outputs); licence assessment `docs/DATASET_LICENSING.md` | yes, run + API + Docker + retraining |

**Things the client must know**

1. MovieLens is licensed for research only. Commercial use needs written permission from
   GroupLens (a negotiated University of Minnesota licence exists) or a commercially licensed /
   client-owned replacement dataset. The MovieTweetings alternative is MIT-licensed by its author
   but needs legal review (tweets, linkable Twitter accounts).
2. Offline, the SVD model's top-10 recommendations score far below a most-popular list
   (MovieLens NDCG@10 0.011 vs 0.147; MovieTweetings 0.0039 vs 0.0568); its strength is rating
   prediction. Validate rankings online.
3. The API has no authentication, TLS or rate limiting; put it behind a gateway (SECURITY.md).
4. The retraining gate (minimum RMSE improvement 0.0, automatic promotion) promoted a
   noise-trained candidate on MovieTweetings; set a stricter policy before production use.

Start: `README.md` → `DEPLOYMENT.md` → `API_DOCUMENTATION.md`.
