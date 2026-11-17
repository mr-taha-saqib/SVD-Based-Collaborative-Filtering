# API documentation: SVD movie recommendation service

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs` (Swagger UI),
`/redoc`, machine-readable schema `/openapi.json`.

All examples below were **captured from the running service** serving the approved
model `svd-k20-fce385dc-c6b9d577` (full transcript:
[`docs/evidence/02_api_curl_examples.txt`](docs/evidence/02_api_curl_examples.txt)).

## Conventions

* JSON in, JSON out. Every response carries an `x-request-id` header (send your own
  `x-request-id` to correlate logs).
* Errors use one shape: `{"error": "<code>", "detail": ..., "request_id": "..."}`.

| Status | Meaning |
|---|---|
| 200 / 202 | success (202: feedback accepted) |
| 404 | `/model/reload` disabled (no `SVD_ADMIN_TOKEN`) |
| 403 | wrong admin token |
| 422 | invalid input (`validation_error`), including `n` above `api.max_n` (100) |
| 503 | no approved model loaded (`model_unavailable`); `/health` also answers 503 |
| 500 | unexpected server error (`internal_error`, logged with the request id) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + readiness (200 only when a model is loaded) |
| GET | `/model/info` | version, k, metrics, data fingerprints, serving policy |
| GET | `/recommendations/{user_id}?n=10&exclude_rated=true` | top-N for a user |
| POST | `/recommendations` | same, with extra `exclude_movie_ids` |
| POST | `/predict` | predicted rating for one (user, movie) |
| POST | `/feedback` | record an observed rating (monitoring + retraining input) |
| GET | `/metrics` | Prometheus text format |
| GET | `/monitoring/summary` | drift, rates, latency, observed performance |
| POST | `/model/reload` | reload the approved model (requires `X-Admin-Token`) |

### Request fields

* `user_id`: integer > 0 (MovieLens `userId`).
* `n`: 1..100 (configurable via `api.max_n`), default 10.
* `exclude_rated`: default `true`; already-rated movies are never returned.
* `exclude_movie_ids`: up to 1,000 ids; ids unknown to the model are reported back
  in `unknown_excluded_movie_ids`, not rejected.
* Unknown fields are rejected.

### Response semantics

* `strategy`: `svd` for known users; `popularity_fallback` (most-rated movies in
  training) for users the model has never seen. `cold_start` mirrors this.
* `score`: SVD = unclipped predicted rating used for ranking; fallback = training
  rating count. `predicted_rating`: clipped to [0.5, 5.0]; `null` for the fallback.
* `title` / `genres`: from MovieLens `movies.csv`, joined on `movieId`;
  `null` / `[]` when the dataset has no entry. Nothing is invented.
* `/predict` `method`: `svd`, `user_bias_only` (unknown movie), `item_bias_only`
  (unknown user) or `global_mean` (both unknown).

## Examples (real responses)

```text
$ curl -s http://localhost:8000/health
{"status":"ok","model_loaded":true,"model_version":"svd-k20-fce385dc-c6b9d577","api_version":"1.0.0","detail":null}
```

```text
$ curl -s "http://localhost:8000/recommendations/1?n=3" | python -m json.tool
{
    "user_id": 1,
    "model_version": "svd-k20-fce385dc-c6b9d577",
    "strategy": "svd",
    "known_user": true,
    "cold_start": false,
    "n_requested": 3,
    "n_returned": 3,
    "items": [
        {
            "rank": 1,
            "movie_id": 3347,
            "title": "Never Cry Wolf (1983)",
            "genres": [
                "Adventure",
                "Drama"
            ],
            "score": 5.355387,
            "predicted_rating": 5.0
        },
        {
            "rank": 2,
            "movie_id": 3213,
            "title": "Batman: Mask of the Phantasm (1993)",
            "genres": [
                "Animation",
                "Children"
            ],
            "score": 5.3459,
            "predicted_rating": 5.0
        },
        {
            "rank": 3,
            "movie_id": 3451,
            "title": "Guess Who's Coming to Dinner (1967)",
            "genres": [
                "Drama"
            ],
            "score": 5.326078,
            "predicted_rating": 5.0
        }
    ],
    "unknown_excluded_movie_ids": [],
    "generated_at": "2026-11-16T22:07:41+00:00"
}
```

```text
$ curl -s -X POST http://localhost:8000/recommendations -H "content-type: application/json" -d '{"user_id": 1, "n": 2, "exclude_movie_ids": [3347, 99999999]}' | python -m json.tool
{
    "user_id": 1,
    "model_version": "svd-k20-fce385dc-c6b9d577",
    "strategy": "svd",
    "known_user": true,
    "cold_start": false,
    "n_requested": 2,
    "n_returned": 2,
    "items": [
        {
            "rank": 1,
            "movie_id": 3213,
            "title": "Batman: Mask of the Phantasm (1993)",
            "genres": [
                "Animation",
                "Children"
            ],
            "score": 5.3459,
            "predicted_rating": 5.0
        },
        {
            "rank": 2,
            "movie_id": 3451,
            "title": "Guess Who's Coming to Dinner (1967)",
            "genres": [
                "Drama"
            ],
            "score": 5.326078,
            "predicted_rating": 5.0
        }
    ],
    "unknown_excluded_movie_ids": [
        99999999
    ],
    "generated_at": "2026-11-16T22:07:41+00:00"
}
```

```text
$ curl -s "http://localhost:8000/recommendations/999999?n=2" | python -m json.tool
{
    "user_id": 999999,
    "model_version": "svd-k20-fce385dc-c6b9d577",
    "strategy": "popularity_fallback",
    "known_user": false,
    "cold_start": true,
    "n_requested": 2,
    "n_returned": 2,
    "items": [
        {
            "rank": 1,
            "movie_id": 356,
            "title": "Forrest Gump (1994)",
            "genres": [
                "Comedy",
                "Drama",
                "Romance",
                "War"
            ],
            "score": 234.0,
            "predicted_rating": null
        },
        {
            "rank": 2,
            "movie_id": 318,
            "title": "Shawshank Redemption, The (1994)",
            "genres": [
                "Crime",
                "Drama"
            ],
            "score": 224.0,
            "predicted_rating": null
        }
    ],
    "unknown_excluded_movie_ids": [],
    "generated_at": "2026-11-16T22:07:41+00:00"
}
```

```text
$ curl -s -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"user_id": 1, "movie_id": 50}'
{"user_id":1,"movie_id":50,"predicted_rating":4.9619,"method":"svd","known_user":true,"known_movie":true,"model_version":"svd-k20-fce385dc-c6b9d577"}

$ curl -s -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"user_id": 1, "movie_id": 99999999}'
{"user_id":1,"movie_id":99999999,"predicted_rating":4.319,"method":"user_bias_only","known_user":true,"known_movie":false,"model_version":"svd-k20-fce385dc-c6b9d577"}

$ curl -s -X POST http://localhost:8000/feedback -H "content-type: application/json" -d '{"user_id": 1, "movie_id": 50, "rating": 4.5}'
{"accepted":true,"stored":true,"model_version":"svd-k20-fce385dc-c6b9d577"}
```

Invalid requests:

```text
$ curl -s -w "\nHTTP %{http_code}" "http://localhost:8000/recommendations/0"
{"error":"validation_error","detail":[{"loc":["path","user_id"],"msg":"Input should be greater than 0","type":"greater_than"}],"request_id":"59b15c289099411a"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" "http://localhost:8000/recommendations/1?n=500"
{"error":"validation_error","detail":"n must be <= 100","request_id":"6c643508e1284646"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/recommendations -H "content-type: application/json" -d '{"n": 5}'
{"error":"validation_error","detail":[{"loc":["body","user_id"],"msg":"Field required","type":"missing"}],"request_id":"f1bf8546f5704085"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/feedback -H "content-type: application/json" -d '{"user_id": 1, "movie_id": 50, "rating": 7}'
{"error":"validation_error","detail":[{"loc":["body","rating"],"msg":"Input should be less than or equal to 5","type":"less_than_equal"}],"request_id":"9141699ea736429f"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/model/reload
{"error":"not_found","detail":"reload endpoint disabled","request_id":"51590dda02cd40d0"}
HTTP 404
```

Model not available (automated test `tests/test_api.py::test_model_unavailable_returns_503`):
`GET /health` returns 503 with `{"status": "unavailable", "model_loaded": false, ...}`
and every model endpoint returns 503 `{"error": "model_unavailable", ...}`.

Prometheus metrics (excerpt):

```text
$ curl -s http://localhost:8000/metrics | grep -E "^recsys_(requests_total|recommendations_total|cold_start_total|model_info)"
recsys_requests_total{endpoint="/health",status="200"} 1.0
recsys_requests_total{endpoint="/model/info",status="200"} 1.0
recsys_requests_total{endpoint="/recommendations/{user_id}",status="200"} 2.0
recsys_requests_total{endpoint="/recommendations",status="200"} 1.0
recsys_requests_total{endpoint="/predict",status="200"} 2.0
recsys_requests_total{endpoint="/feedback",status="202"} 1.0
recsys_requests_total{endpoint="/recommendations/{user_id}",status="422"} 2.0
recsys_requests_total{endpoint="/recommendations",status="422"} 1.0
recsys_requests_total{endpoint="/feedback",status="422"} 1.0
recsys_requests_total{endpoint="/model/reload",status="404"} 1.0
recsys_recommendations_total{strategy="svd"} 2.0
recsys_recommendations_total{strategy="popularity_fallback"} 1.0
recsys_cold_start_total 1.0
recsys_model_info{version="svd-k20-fce385dc-c6b9d577"} 1.0
```

Monitoring summary:

```text
$ curl -s http://localhost:8000/monitoring/summary   (after 4 recommendation requests incl. 1 unknown user, 1 invalid request)
{
  "model_version": "svd-k20-fce385dc-c6b9d577",
  "uptime_seconds": 5.5,
  "window_size": 5000,
  "requests_in_window": 4,
  "totals": {
    "recommendation_requests": 4,
    "items_recommended": 12,
    "requests": 5,
    "cold_start_fallback": 1,
    "unknown_user": 1,
    "error:validation": 1,
    "client_errors": 1
  },
  "latency_ms": {
    "p50": 1.887,
    "p95": 4.84,
    "p99": 5.369
  },
  "rates": {
    "cold_start_fallback": 0.25,
    "unknown_user": 0.25,
    "short_response": 0.0
  },
  "recommendation_distribution": {
    "distinct_items_recommended": 6,
    "popularity_decile_counts": [
      0,
      0,
      0,
      0,
      0,
      0,
      3,
      6,
      0,
      0
    ],
    "reference_popularity_distribution": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.23426229508196722,
      0.6478688524590164,
      0.1039344262295082,
      0.013934426229508197
    ]
  },
  "input_drift": {
    "signal": "requesting-user activity profile vs training users (PSI)",
    "psi": 5.9009,
    "status": "insufficient_data",
    "observed_counts": {
      "0-19": 0,
      "20-49": 2,
      "50-99": 0,
      "100-199": 1,
      "200-499": 0,
      "500-inf": 0,
      "unknown_user": 1
    }
  },
  "prediction_drift": {
    "signal": "popularity deciles of recommended items vs build-time reference (PSI)",
    "psi": 0.825,
    "status": "insufficient_data",
    "based_on_svd_requests": 3
  },
  "observed_performance": {
    "status": "insufficient_labels",
    "n_labelled": 0,
    "required": 30,
    "note": "No performance claim is made without enough labelled feedback."
  },
  "data_quality": {
    "client_errors": 1,
    "validation_errors": 1,
    "unknown_movie_ids_in_requests": 0
  },
  "thresholds": {
    "psi_warning": 0.1,
    "psi_alert": 0.25,
    "min_requests_for_drift": 50,
    "min_feedback_for_performance": 30
  }
}
{
    "model_version": "svd-k20-fce385dc-c6b9d577",
    "uptime_seconds": 5.0,
    "window_size": 5000,
    "requests_in_window": 3,
    "totals": {
        "requests": 13,
        "recommendation_requests": 3,
        "items_recommended": 7,
        "unknown_movie_ids_in_requests": 1,
        "cold_start_fallback": 1,
        "unknown_user": 1,
        "feedback": 1,
        "error:validation": 3,
        "client_errors": 5
    },
    "latency_ms": {
        "p50": 1.127,
        "p95": 6.735,
        "p99": 12.141
    },
    "rates": {
        "cold_start_fallback": 0.3333,
        "unknown_user": 0.3333,
        "short_response": 0.0
    },
    "recommendation_distribution": {
        "distinct_items_recommended": 5,
        "popularity_decile_counts": [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            4,
            0,
            0
        ],
        "reference_popularity_distribution": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.23426229508196722,
            0.6478688524590164,
            0.1039344262295082,
            0.013934426229508197
        ]
    },
    "input_drift": {
        "signal": "requesting-user activity profile vs training users (PSI)",
        "psi": 10.0431,
        "status": "insufficient_data",
        "observed_counts": {
            "0-19": 0,
            "20-49": 0,
```

## Python client example

```python
import httpx
with httpx.Client(base_url="http://localhost:8000", timeout=5) as c:
    recs = c.post("/recommendations", json={"user_id": 42, "n": 5}).json()
    for item in recs["items"]:
        print(item["rank"], item["movie_id"], item["title"], item["predicted_rating"])
```
