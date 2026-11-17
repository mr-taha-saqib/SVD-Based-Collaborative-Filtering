# Security and limitations

Release `v1.0.0-mlops`. Summary for decision makers; the detailed review with evidence is
[SECURITY.md](SECURITY.md). **Status: delivery-ready reference implementation, not production-ready.**

## 1. Current security measures (implemented and tested)

| Measure | Detail |
|---|---|
| Input validation | strict schemas (`extra="forbid"`), ids > 0, `n` ≤ 100, ratings 0.5–5.0, ≤ 1,000 exclusions; HTTP 422 with field-level messages |
| Pseudonymised monitoring | the event log stores a salted SHA-256 pseudonym of the user, never the raw id; an unset or placeholder salt is replaced by a random per-process salt |
| Model integrity | `model.npz` SHA-256 verified before serving; arrays loaded with `allow_pickle=False` |
| Dataset integrity | MovieLens archive SHA-256 pinned; path-traversal-safe extraction; MovieTweetings files checksum-pinned |
| Admin endpoint | `/model/reload` disabled unless `SVD_ADMIN_TOKEN` is set; constant-time comparison |
| Container | non-root uid 10001, code not writable, registry volume mounted read-only, `cap_drop: ALL`, `no-new-privileges`, health check, memory limit |
| Network exposure | Compose publishes the API and MLflow on **127.0.0.1 only** (`API_BIND`) |
| Dependencies | pinned; pip-audit clean for the project environment and both images |
| Secrets | none in the repository or images; `.env` git-ignored; `.env.example` placeholders only |

## 2. Controls you must provide

| Control | Status | Required action |
|---|---|---|
| Authentication / authorisation | **not implemented**: any caller can request any user's recommendations or post feedback | gateway with authentication; per-user authorisation (users see only their own lists); `/feedback` only from your backend (feedback drives retraining); `/metrics` internal |
| TLS | **not implemented** | terminate TLS at the proxy |
| Rate limiting | **not implemented** | per-client limits at the proxy |
| Request-size limit | schema limits only | body-size limit at the proxy |
| Secret management | environment / `.env` | secret manager for `SVD_MONITORING_SALT`, `SVD_ADMIN_TOKEN` |
| Registry and MLflow access | file permissions; MLflow without authentication (localhost) | restrict write access to the release operator; authenticated MLflow if shared |
| Retention | `data/feedback/api_feedback.csv` and `monitoring/events.jsonl` grow without bound | retention, deletion and rotation policy |
| Promotion policy | automatic promotion when the gate passes; `min_rmse_improvement` 0.0 | positive margin and human approval before production |

## 3. Data privacy

* `POST /feedback` stores **raw user ids, movie ids and ratings** (needed for retraining). For real
  users these are personal data (GDPR): define a legal basis, retention and deletion.
* The monitoring event log contains only pseudonyms and aggregates.
* MovieTweetings ratings are linkable to real Twitter accounts through the public `users.dat`
  (not downloaded by this project); they are pseudonymous personal data.

## 4. Dataset licensing (summary)

* **MovieLens: research use only.** Commercial or revenue-bearing use requires written permission;
  the University of Minnesota offers a negotiated commercial licence (Technology ID z05173). No
  permission has been obtained. Redistribution of the data is not allowed, so it is not included.
* **MovieTweetings:** MIT-licensed by its author; third-party terms (X/Twitter content, IMDb
  references) and GDPR questions require the client's legal review.
* Details: [docs/DATASET_LICENSING.md](docs/DATASET_LICENSING.md), [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md).

## 5. Known risks and model limitations

* Offline top-10 ranking: most popular (NDCG@10 0.147) far ahead of SVD (0.011). SVD's strength is
  rating prediction (test RMSE 0.8853 vs 0.8943 bias baseline).
* Retraining was demonstrated with synthetic feedback only.
* With the default gate, a MovieTweetings candidate trained on random feedback was promoted
  because it tied the current model.
* Random per-user split (not temporal); popularity-only cold start; dense matrix memory growth.

## 6. Production deployment requirements

A cleared dataset (MovieLens commercial licence, MovieTweetings legal clearance, or the client's own
data); gateway with TLS, authentication, authorisation, rate limiting; secret management;
retention policy; stricter promotion policy with human approval; online (A/B) validation of
recommendation quality; monitoring/alerting integration; volume backups.

## 7. Outside the delivered scope

Cloud deployment, user management, a front end, real-time incremental updates, temporal
evaluation, hybrid/content-based models, legal review and dataset licensing.
