# Monitoring

Implemented in `src/monitoring.py`, exposed by the API, verified in
`docs/evidence/05_monitoring_drift_test.log`.

## What is tracked

| Signal | Where | Notes |
|---|---|---|
| Requests by endpoint and status, latency histogram | `/metrics` | Prometheus text format |
| Recommendations by strategy, items recommended, cold starts | `/metrics`, summary | `strategy` = svd / popularity_fallback |
| Errors by type (validation, internal) | `/metrics`, summary | |
| Loaded model version | `recsys_model_info{version=...}` | |
| Unknown-user and cold-start rates, short responses | summary `rates` | |
| Recommendation popularity distribution | summary | deciles of training rating counts |
| **Input drift** | summary `input_drift` | PSI of requesting-user activity buckets vs training users |
| **Prediction drift** | summary `prediction_drift` | PSI of recommended-item popularity deciles vs build-time reference (SVD responses only) |
| **Observed performance** | summary `observed_performance` | RMSE/MAE vs ratings from `POST /feedback`; shown only with ≥ 30 labels |
| Data quality | summary `data_quality` | validation errors, unknown movie ids in requests |

PSI status: `ok` < 0.10 ≤ `warning` < 0.25 ≤ `alert`; `insufficient_data` below 50
requests in the window (all configurable in `config.yaml` → `monitoring`).

Drift is never reported as performance degradation; performance is reported only
from delayed ground truth.

## Event log

`monitoring/events.jsonl`, one JSON object per recommendation or feedback event:
timestamp, model version, **pseudonymised user** (salted SHA-256, salt from
`SVD_MONITORING_SALT`), activity bucket, strategy, n requested/returned, popularity
deciles, latency. Offline report:

```bash
python src/monitoring.py --log monitoring/events.jsonl --out monitoring_report.json
```

## Controlled drift test (executed)

| Phase | Traffic | Input drift | Prediction drift | Cold start |
|---|---|---|---|---|
| 1 | 300 requests, known users sampled uniformly | PSI 0.0078, ok | PSI 0.0006, ok | 0 % |
| 1b | + 40 random synthetic feedback ratings | – | – | performance "available", RMSE 2.10 (random labels, as expected) |
| 2 | 300 requests, unknown user ids only (fresh process) | PSI 16.78, **alert** | insufficient_data (no SVD responses) | 100 % |

Reproduce: start the API, then
`python scripts/simulate_traffic.py --url http://localhost:8000 --mode normal --n 300`
(or `--mode drift`, `--mode feedback`).

## Operating notes

* Metrics live in process memory (window of 5,000 requests) and reset on restart; the
  event log persists. Scrape `/metrics` with Prometheus for history.
* The activity reference assumes request users resemble training users. Real traffic
  skews to active users; recalibrate the reference from production logs before
  alerting on input drift.
