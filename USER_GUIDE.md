# User guide

How to use the recommender. Examples use MovieLens ids (public research data). Endpoint details:
[API_REFERENCE.md](API_REFERENCE.md); setup: [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md).

## 1. Typical workflow

1. Train and approve a model (`python src/main.py --approve ...` or the Docker trainer).
2. Your application asks `GET /recommendations/{user_id}` for a user's list, or `POST /predict`
   for one user–movie pair.
3. When a user actually rates a movie, send it to `POST /feedback`; it measures live error and
   collects data for retraining.
4. Watch `GET /monitoring/summary` / `GET /metrics` (drift, cold-start rate, observed RMSE).
5. Periodically run gated retraining on the collected feedback and review the report.

## 2. Inputs

| Input | Rule |
|---|---|
| `user_id`, `movie_id` | integers > 0 (MovieLens ids for the delivered model) |
| `n` | 1–100 recommendations (`api.max_n`; the schema accepts up to 1,000, the service rejects > 100) |
| `exclude_rated` | default `true`: never recommend movies the user already rated in training |
| `exclude_movie_ids` | up to 1,000 extra ids to exclude (unknown ids are reported back) |
| `rating` (feedback) | 0.5–5.0 |

## 3. Outputs

* Recommendations: `strategy` is `svd` for known users or `popularity_fallback` for unknown users
  (cold start); each item has `rank`, `movie_id`, `title`, `genres`, `score`, `predicted_rating`
  (SVD only).
* Prediction: `predicted_rating` (clipped to 0.5–5.0) and `method` (`svd`, `user_bias_only`,
  `item_bias_only`, `global_mean`).

## 4. Practical examples

```bash
curl -s "http://127.0.0.1:8000/recommendations/1?n=5"
curl -s "http://127.0.0.1:8000/recommendations/999999?n=3"           # unknown user -> popular movies
curl -s -X POST http://127.0.0.1:8000/recommendations -H 'content-type: application/json' \
     -d '{"user_id": 1, "n": 3, "exclude_rated": true, "exclude_movie_ids": [3347]}'
curl -s -X POST http://127.0.0.1:8000/predict  -H 'content-type: application/json' -d '{"user_id": 1, "movie_id": 50}'
curl -s -X POST http://127.0.0.1:8000/feedback -H 'content-type: application/json' -d '{"user_id": 1, "movie_id": 50, "rating": 4.5}'
./examples/demo/run_demo.sh                                           # all endpoints, readable output
```

Captured output: `examples/demo/demo_output_example.txt`. In that run user 1's top-5 list is all
predicted at 5.0 (clipped) and consists of less-known films; that is typical of SVD on sparse data
and one reason the most-popular baseline ranks better offline.

## 5. Choosing between rating prediction and recommendations

* Use `POST /predict` when you need an estimated rating (e.g. "you will probably rate this 4.2").
  This is where the model is strongest (test RMSE 0.8853 vs 0.8943 for the best simple baseline).
* Use `/recommendations` for lists, but compare with a popularity list in an online test: offline,
  most-popular scores NDCG@10 0.147 vs 0.011 for SVD.

## 6. Error handling

| HTTP | `error` | Typical cause |
|---|---|---|
| 422 | `validation_error` | id ≤ 0 or not an integer, `n` > 100, rating outside 0.5–5.0, unknown field, malformed JSON |
| 404 | `not_found` | `/model/reload` while `SVD_ADMIN_TOKEN` is unset |
| 403 | `forbidden` | wrong `X-Admin-Token` |
| 503 | `model_unavailable` | no approved model or checksum mismatch (`/health` shows why) |
| 500 | `internal_error` | unexpected; logged with `request_id` |

Unknown users and movies are **not** errors: they get the fallback strategy / method.

## 7. Model management

```bash
python -c "import sys; sys.path.insert(0,'src'); from registry import ModelRegistry as R; r=R('models'); print('approved:', r.approved_version()); print(r.list_versions())"
curl -s http://127.0.0.1:8000/model/info | python -m json.tool
python src/retrain.py --rollback --reason "regression reported"      # re-approve the previous version
```

A new approved version is served after an API restart, or immediately via
`curl -X POST -H "X-Admin-Token: $SVD_ADMIN_TOKEN" http://127.0.0.1:8000/model/reload`.
Every decision is recorded with a reason in `models/registry.json`; MLflow's `champion` alias follows.

## 8. Monitoring

`GET /monitoring/summary`: request counts, unknown-user and cold-start rates, input drift (who is
asking: user-activity buckets vs training), prediction drift (popularity of what is recommended),
observed RMSE/MAE from feedback (from 30 labels), latency, data-quality counters.
`GET /metrics`: Prometheus `recsys_*` metrics. Try it: `python scripts/simulate_traffic.py --url
http://127.0.0.1:8000 --mode normal --n 200`, then restart the API and use `--mode drift`.
User ids are pseudonymised in the event log with a salted hash; set `SVD_MONITORING_SALT`.

## 9. Retraining

```bash
# real use: the API writes POST /feedback ratings to data/feedback/api_feedback.csv
python src/retrain.py --feedback data/feedback/api_feedback.csv --dry-run   # evaluate, change nothing
python src/retrain.py --feedback data/feedback/api_feedback.csv             # promote only if the gate passes
# practice with SYNTHETIC feedback
python scripts/generate_synthetic_feedback.py --mode new_users --n 3000 --out /tmp/fb.csv
```

Requirements: ≥ 100 new ratings from ≥ 5 users. The report (`reports/retraining/`) shows the
candidate and approved metrics on the same validation data and the gate decision. **Before
production**, raise `retraining.promotion.min_rmse_improvement` above 0 and add a human approval
step: with the default a candidate that only ties the current model is promoted.

## 10. Alternative dataset (MovieTweetings)

`python src/run_movietweetings.py` trains a separate model in `models_external/movietweetings/`
(results in [docs/MOVIETWEETINGS_EVALUATION.md](docs/MOVIETWEETINGS_EVALUATION.md)). Use it only after
the client's legal review ([docs/DATASET_LICENSING.md](docs/DATASET_LICENSING.md)).
