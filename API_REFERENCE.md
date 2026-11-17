# API reference

Implemented in `src/api.py` (API version `1.0.0`, release `v1.0.0-mlops`). Machine-readable schema:
[`docs/openapi.json`](docs/openapi.json) (exported from the running service); interactive docs at
`/docs` and `/redoc`. Examples are **real responses** captured from the service serving
`svd-k20-fce385dc-c6b9d577` (MovieLens latest-small); see `examples/demo/demo_output_example.txt`
and `docs/evidence/02_api_curl_examples.txt`.

Base URL: `http://127.0.0.1:8000`. Content type: `application/json`.

## Authentication

**None** on recommendation, prediction, feedback and monitoring endpoints. Only
`POST /model/reload` requires `X-Admin-Token` and is disabled unless `SVD_ADMIN_TOKEN` is set.
Provide TLS, authentication, per-user authorisation and rate limiting at a gateway
([SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md)).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model loaded and checksum verified |
| GET | `/model/info` | version, algorithm, k, users/items, serving policy, training data, metrics, limits |
| GET | `/recommendations/{user_id}` | top-n recommendations |
| POST | `/recommendations` | top-n with extra exclusions |
| POST | `/predict` | predicted rating for a user–movie pair |
| POST | `/feedback` | observed rating (delayed ground truth; appended for retraining) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/monitoring/summary` | monitoring snapshot |
| POST | `/model/reload` | reload the approved model (admin) |

## Error format

`{"error": "<code>", "detail": <string or list>, "request_id": "<id>"}`. Codes: `validation_error`
(422), `not_found` (404), `forbidden` (403), `model_unavailable` (503), `internal_error` (500).
Unknown users or movies are handled by fallbacks, not errors.

---

### GET /health

`{status: "ok"|"unavailable", model_loaded, model_version, api_version, detail}`; 200 or 503.

```text
$ curl -s http://127.0.0.1:8000/health
{"status": "ok", "model_loaded": true, "model_version": "svd-k20-fce385dc-c6b9d577", "api_version": "1.0.0", "detail": null}
```

### GET /model/info

Keys: `model_version`, `model_name`, `algorithm`, `k`, `n_users`, `n_items`, `serving_policy`,
`training_data` (dataset and split SHA-256, `trained_on`, `n_train_ratings`), `metrics`,
`has_movie_metadata`, `limits`, `mlflow_run_id`, `git_commit`, `created_at`, `status`. Excerpt:

```json
{"model_version": "svd-k20-fce385dc-c6b9d577", "algorithm": "truncated SVD with additive bias decomposition",
 "k": 20, "n_users": 610, "n_items": 8578,
 "serving_policy": {"min_item_ratings": 5, "exclude_rated": true, "cold_start": "most-popular items by training rating count"},
 "limits": {"default_n": 10, "max_n": 100}}
```

### GET /recommendations/{user_id}

| Parameter | In | Type | Rules |
|---|---|---|---|
| `user_id` | path | int | > 0 |
| `n` | query | int | 1–1,000 by schema; > `api.max_n` (100) → 422; default 10 |
| `exclude_rated` | query | bool | default `true` |

Response `RecommendationResponse`: `user_id`, `model_version`, `strategy` (`svd` |
`popularity_fallback`), `known_user`, `cold_start`, `n_requested`, `n_returned`, `items` (list of
`{rank, movie_id, title, genres, score, predicted_rating}`), `unknown_excluded_movie_ids`,
`generated_at`. `score` is the unclipped predicted rating (SVD) or the training rating count
(fallback); `predicted_rating` is clipped to 0.5–5.0 (SVD only, else null).

Captured (summarised):

```text
GET /recommendations/1?n=5        -> strategy svd, known_user true, 5/5
   1. Never Cry Wolf (1983)  predicted 5.00
   2. Batman: Mask of the Phantasm (1993)  predicted 5.00
   3. Guess Who's Coming to Dinner (1967)  predicted 5.00
   4. Paths of Glory (1957)  predicted 5.00
   5. American Splendor (2003)  predicted 5.00
GET /recommendations/999999?n=3   -> strategy popularity_fallback, known_user false
   1. Forrest Gump (1994)   2. Shawshank Redemption, The (1994)   3. Pulp Fiction (1994)
```

Errors: `/recommendations/0` → 422 `{"error":"validation_error","detail":[{"loc":["path","user_id"],"msg":"Input should be greater than 0","type":"greater_than"}],...}`;
`n=5000` → 422 (`less_than_equal`, 1000); `n=500` → 422 `"n must be <= 100"`.

### POST /recommendations

Body `RecommendationRequest` (unknown fields rejected): `user_id` (> 0), `n` (1–1,000; > 100 → 422;
default 10), `exclude_rated` (default true), `exclude_movie_ids` (≤ 1,000 ints). Response as above.

```text
$ curl -s -X POST http://127.0.0.1:8000/recommendations -H 'content-type: application/json' \
    -d '{"user_id": 1, "n": 3, "exclude_rated": true, "exclude_movie_ids": [3347, 99999999]}'
-> strategy svd, 3/3: Batman: Mask of the Phantasm (1993), Guess Who's Coming to Dinner (1967),
   Paths of Glory (1957); "unknown_excluded_movie_ids": [99999999]
```

### POST /predict

Body `PredictionRequest`: `user_id` (> 0), `movie_id` (> 0). Response `PredictionResponse`:
`user_id`, `movie_id`, `predicted_rating`, `method` (`svd` | `user_bias_only` | `item_bias_only` |
`global_mean`), `known_user`, `known_movie`, `model_version`.

```text
{"user_id": 1, "movie_id": 50}            -> {"user_id": 1, "movie_id": 50, "predicted_rating": 4.9619, "method": "svd", "known_user": true, "known_movie": true, "model_version": "svd-k20-fce385dc-c6b9d577"}
{"user_id": 999999, "movie_id": 99999999} -> {"predicted_rating": 3.5009, "method": "global_mean", "known_user": false, "known_movie": false, ...}
{"user_id": "abc", "movie_id": 50}        -> 422 {"error":"validation_error","detail":[{"loc":["body","user_id"],"msg":"Input should be a valid integer, unable to parse string as an integer","type":"int_parsing"}],...}
```

### POST /feedback

Body `FeedbackRequest`: `user_id` (> 0), `movie_id` (> 0), `rating` (0.5–5.0). HTTP **202**.
Response `{accepted, stored, model_version}`; `stored` is true when the rating was appended to
`SVD_FEEDBACK_PATH` (CSV `userId,movieId,rating,timestamp`, personal data: apply a retention policy).

```text
{"user_id": 1, "movie_id": 50, "rating": 4.5} -> 202 {"accepted": true, "stored": true, "model_version": "svd-k20-fce385dc-c6b9d577"}
{"user_id": 1, "movie_id": 50, "rating": 6.0} -> 422 {"error":"validation_error","detail":[{"loc":["body","rating"],"msg":"Input should be less than or equal to 5","type":"less_than_equal"}],...}
```

### GET /metrics

Prometheus text. Families: `recsys_requests_total{endpoint,status}`, `recsys_request_latency_seconds`,
`recsys_recommendations_total`, `recsys_recommended_items_total`, `recsys_cold_start_total`,
`recsys_feedback_total`, `recsys_errors_total`, `recsys_drift_psi`, `recsys_model_info`.

```text
recsys_requests_total{endpoint="/recommendations/{user_id}",status="200"} 2.0
recsys_requests_total{endpoint="/recommendations/{user_id}",status="422"} 1.0
recsys_requests_total{endpoint="/feedback",status="202"} 1.0
```

### GET /monitoring/summary

Keys: `model_version`, `uptime_seconds`, `window_size`, `requests_in_window`, `totals`, `rates`
(`cold_start_fallback`, `unknown_user`, `short_response`), `input_drift`, `prediction_drift`
(`status`, `psi`), `recommendation_distribution`, `observed_performance` (from feedback; ≥ 30
labels), `latency_ms`, `data_quality`, `thresholds`.

```json
{"model_version": "svd-k20-fce385dc-c6b9d577", "requests_in_window": 3,
 "rates": {"cold_start_fallback": 0.3333, "unknown_user": 0.3333, "short_response": 0.0},
 "input_drift": {"status": "insufficient_data"},
 "observed_performance": {"status": "insufficient_labels", "n_labelled": 1, "required": 30,
                          "note": "No performance claim is made without enough labelled feedback."}}
```

### POST /model/reload

Header `X-Admin-Token`. Disabled → 404 `{"error":"not_found","detail":"reload endpoint disabled"}`;
wrong/missing token → 403 `{"error":"forbidden","detail":"invalid admin token"}`.
