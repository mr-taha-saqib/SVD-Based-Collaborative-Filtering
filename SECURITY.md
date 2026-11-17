# Security and production-readiness review

Review date: 2026-11-17 (session 3). Scope: the FastAPI recommendation service, Docker images and
Compose file, model registry, monitoring, retraining tooling and dependencies. **Result: the
service is a reference implementation, not production-ready.** Items marked *prerequisite* must
be in place before the API is exposed beyond a trusted host.

## Summary

| # | Area | Finding | Status |
|---|---|---|---|
| 1 | Authentication / authorisation | `/recommendations`, `/predict`, `/feedback`, `/metrics`, `/monitoring/summary`, `/model/info` are unauthenticated; any caller can request recommendations for any user id. Only `/model/reload` is protected (`X-Admin-Token`, disabled unless `SVD_ADMIN_TOKEN` is set) | **Prerequisite**: authenticate clients at a gateway / reverse proxy and authorise per user (a user should only see their own recommendations); restrict `/feedback` to the application backend (unauthenticated ratings could poison retraining); keep `/metrics` internal |
| 2 | Admin token comparison | was a plain `!=` comparison (timing side channel) | **Fixed**: `hmac.compare_digest`; tests for missing and near-miss tokens |
| 3 | Pseudonymisation salt | with `SVD_MONITORING_SALT` unset the monitor hashed user ids **without a salt**; the `.env.example` placeholder value was public. User ids are small integers, so both made the pseudonyms trivially reversible | **Fixed**: an unset or placeholder salt is replaced by a random per-process salt (warning logged; pseudonyms then change on restart). Set a secret salt for stable pseudonyms. Tests added |
| 4 | Input validation | Pydantic: ids > 0, `n` 1–1,000 (capped by `api.max_n`), ≤ 1,000 excluded ids, ratings 0.5–5.0; verified 422 for user 0, malformed bodies and out-of-range ratings | OK. **Prerequisite**: request-body size limit at the proxy |
| 5 | Rate limiting | none in the application | **Prerequisite**: per-client limits at the proxy |
| 6 | TLS | Uvicorn serves plain HTTP | **Prerequisite**: terminate TLS at the proxy. Compose now publishes the API on **127.0.0.1 only** by default (`API_BIND`); MLflow likewise |
| 7 | Secrets | none in the repository or images (scans in `FINAL_E2E_ACCEPTANCE.md`); `.env` is git-ignored; the build CA is a BuildKit secret; the salt and admin token come from the environment | OK. Recommended: a secret manager |
| 8 | Container | non-root uid 10001; the serving image previously made `/app` (including code) owned by that user | **Fixed**: only `models/`, `monitoring/`, `data/` are writable by uid 10001; the API now mounts the registry **read-only**; Compose adds `cap_drop: ALL` and `no-new-privileges`. Verified in `docs/evidence/12_session3_docker_hardened_compose.log` |
| 9 | Registry permissions | version directories were created with mode 0700 (inherited from the temporary build directory), so a registry trained by one user could not be served read-only by the uid-10001 container | **Fixed** in `src/registry.py` (0755, like the files); regression test added |
| 10 | Dependencies | pip-audit (2026-11-17): project dependencies clean; serving and trainer images had pip 25.0.1 / setuptools 78.1.0 from the base image with published advisories | **Fixed**: images upgrade pip 26.3 / setuptools 84.2.1; rebuilt images audit clean. Recommended: pip-audit in CI, base image pinned by digest |
| 11 | Privacy / data retention | the event log stores salted pseudonyms, activity buckets and popularity bins only. **`/feedback` appends raw user ids, movie ids and ratings to `data/feedback/api_feedback.csv`** (needed for retraining; personal data under the GDPR) | **Prerequisite**: retention and deletion policy for feedback; rotation for `monitoring/events.jsonl` (grows without bound) |
| 12 | Logging | access log: request id, method, endpoint, status, latency; no bodies, no user ids | OK |
| 13 | Model / registry access | SHA-256 of `model.npz` verified at load; approval, promotion and rollback are CLI operations without user authentication | **Prerequisite**: restrict registry write access to the release operator |
| 14 | Retraining gate | `min_rmse_improvement` is 0.0 and promotion is automatic once the gate passes. On MovieTweetings a candidate trained on random-noise feedback tied the approved model and **was promoted** (`docs/MOVIETWEETINGS_EVALUATION.md`) | **Client decision**: set a positive minimum improvement and/or require human approval before promotion in production |
| 15 | MLflow | no authentication | Compose binds it to 127.0.0.1. **Prerequisite** for shared use: an authenticated tracking server |
| 16 | Data licence | MovieLens is research-only; MovieTweetings needs legal review (`docs/DATASET_LICENSING.md`) | **Client decision** |

## Reference deployment shape

```
client ──HTTPS──> reverse proxy / API gateway  (TLS, authN/authZ, rate limit, body-size limit)
                     └──HTTP (private network)──> svd API container (uid 10001, read-only registry)
application backend ──> /feedback (authenticated service identity only)
operators ──> trainer / retrain CLI ──> registry volume (write access restricted)
```

## Reporting a vulnerability

Report privately to the repository owner (GitHub: mr-taha-saqib). Do not open a public issue
with exploit details.
