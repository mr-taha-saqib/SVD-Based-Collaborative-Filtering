"""
monitoring.py
=============
Operational and model monitoring for the recommendation API.

Three kinds of signal are kept strictly apart:

* **Input drift**: the activity profile of requesting users (training-rating
  buckets, plus "unknown_user") compared with the training reference via PSI.
* **Prediction drift**: the popularity deciles of recommended items compared with
  the distribution the model produced for training users at build time (PSI).
* **Observed performance**: RMSE/MAE of the model's predictions against ratings
  that arrive later through ``POST /feedback`` (delayed ground truth). Reported
  only when enough labelled feedback exists; drift alone is never reported as
  performance degradation.

Prometheus metrics are exposed through a private ``CollectorRegistry`` (one per
app instance). Events are appended to a JSONL log with pseudonymised user ids
(salted SHA-256), never raw ids.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Histogram
from prometheus_client import Counter as PCounter

logger = logging.getLogger(__name__)

EPS = 1e-4


def psi(expected: list[float] | np.ndarray, actual: list[float] | np.ndarray) -> float:
    """Population Stability Index between two discrete distributions."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.sum() <= 0 or a.sum() <= 0:
        return float("nan")
    e = np.clip(e / e.sum(), EPS, None)
    a = np.clip(a / a.sum(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_status(value: float, n: int, min_n: int, warn: float, alert: float) -> str:
    if n < min_n or math.isnan(value):
        return "insufficient_data"
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warning"
    return "ok"



# Values that must never be used as the pseudonymisation salt (published defaults).
PLACEHOLDER_SALTS = frozenset({"change-me-to-a-random-string", "changeme", "change-me"})

class RecommendationMonitor:
    """In-process monitor; one instance per API process."""

    def __init__(self, reference: dict[str, Any] | None, cfg: dict[str, Any],
                 model_version: str, log_dir: Path | None = None):
        self.reference = reference or {}
        self.window = int(cfg.get("window_size", 5000))
        self.psi_warn = float(cfg.get("psi_warning", 0.10))
        self.psi_alert = float(cfg.get("psi_alert", 0.25))
        self.min_requests = int(cfg.get("min_requests_for_drift", 50))
        self.min_feedback = int(cfg.get("min_feedback_for_performance", 30))
        salt = os.environ.get(str(cfg.get("hash_salt_env", "SVD_MONITORING_SALT")), "")
        if not salt or salt in PLACEHOLDER_SALTS:
            # User ids are small integers: an unsalted (or publicly known) salt would make
            # the pseudonyms trivially reversible. Use a random per-process salt instead;
            # pseudonyms are then not comparable across restarts.
            logger.warning("No monitoring salt configured (or the .env.example placeholder is "
                           "still set); using a random per-process salt. Set SVD_MONITORING_SALT "
                           "to a secret value to keep pseudonyms stable across restarts.")
            salt = secrets.token_hex(16)
        self.salt = salt
        self.model_version = model_version
        self.log_path = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log_path = Path(log_dir) / "events.jsonl"
        self._lock = threading.Lock()
        self.started_at = time.time()

        self.activity_labels: list[str] = list(self.reference.get("user_activity_labels", []))
        self.item_pop_bins = np.asarray(self.reference.get("item_pop_bins", []), dtype=np.int64)
        self.n_bins = len(self.reference.get("recommended_popularity_distribution", [])) or 10

        self._requests: deque[dict[str, Any]] = deque(maxlen=self.window)
        self._pop_bins: deque[np.ndarray] = deque(maxlen=self.window)
        self._feedback_err: deque[float] = deque(maxlen=self.window)
        self._latencies: deque[float] = deque(maxlen=self.window)
        self._recommended_items: Counter[int] = Counter()
        self.totals: Counter[str] = Counter()

        self.registry = CollectorRegistry()
        self.m_requests = PCounter("recsys_requests_total", "API requests", ["endpoint", "status"],
                                   registry=self.registry)
        self.m_latency = Histogram("recsys_request_latency_seconds", "Request latency", ["endpoint"],
                                   buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
                                   registry=self.registry)
        self.m_recs = PCounter("recsys_recommendations_total", "Recommendation responses by strategy",
                               ["strategy"], registry=self.registry)
        self.m_items = PCounter("recsys_recommended_items_total", "Items recommended", registry=self.registry)
        self.m_cold = PCounter("recsys_cold_start_total", "Popularity-fallback responses (unknown user)",
                               registry=self.registry)
        self.m_errors = PCounter("recsys_errors_total", "Errors by type", ["type"], registry=self.registry)
        self.m_feedback = PCounter("recsys_feedback_total", "Feedback ratings received", registry=self.registry)
        self.m_model = Gauge("recsys_model_info", "Loaded model version", ["version"], registry=self.registry)
        self.m_psi = Gauge("recsys_drift_psi", "Latest PSI value", ["signal"], registry=self.registry)
        self.m_model.labels(version=model_version).set(1)

    # ── helpers ───────────────────────────────────────────────────────────────
    def pseudonymise(self, user_id: int) -> str:
        return hashlib.sha256(f"{self.salt}:{int(user_id)}".encode()).hexdigest()[:16]

    def _append(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    # ── recording ─────────────────────────────────────────────────────────────
    def record_request(self, endpoint: str, status: int, latency_s: float) -> None:
        self.m_requests.labels(endpoint=endpoint, status=str(status)).inc()
        self.m_latency.labels(endpoint=endpoint).observe(latency_s)
        with self._lock:
            self.totals["requests"] += 1
            self._latencies.append(latency_s)
            if status >= 500:
                self.totals["server_errors"] += 1
            elif status >= 400:
                self.totals["client_errors"] += 1

    def record_error(self, kind: str) -> None:
        self.m_errors.labels(type=kind).inc()
        with self._lock:
            self.totals[f"error:{kind}"] += 1

    def record_recommendation(self, user_id: int, activity: str, strategy: str,
                              item_cols: list[int], n_requested: int,
                              n_unknown_excluded: int, latency_s: float) -> None:
        self.m_recs.labels(strategy=strategy).inc()
        self.m_items.inc(len(item_cols))
        if strategy == "popularity_fallback":
            self.m_cold.inc()
        bins = (self.item_pop_bins[np.asarray(item_cols, dtype=np.int64)]
                if len(self.item_pop_bins) and item_cols else np.empty(0, dtype=np.int64))
        with self._lock:
            self.totals["recommendation_requests"] += 1
            self.totals["items_recommended"] += len(item_cols)
            if strategy == "popularity_fallback":
                self.totals["cold_start_fallback"] += 1
            if activity == "unknown_user":
                self.totals["unknown_user"] += 1
            if n_unknown_excluded:
                self.totals["unknown_movie_ids_in_requests"] += n_unknown_excluded
            if len(item_cols) < n_requested:
                self.totals["short_responses"] += 1
            self._requests.append({"activity": activity, "strategy": strategy})
            if strategy == "svd":          # prediction drift concerns model output only
                self._pop_bins.append(bins)
            self._recommended_items.update(int(c) for c in item_cols)
        self._append({"ts": self._now(), "event": "recommendation", "model_version": self.model_version,
                      "user": self.pseudonymise(user_id), "activity": activity, "strategy": strategy,
                      "n_requested": n_requested, "n_returned": len(item_cols),
                      "popularity_bins": bins.tolist(), "latency_ms": round(latency_s * 1000, 3)})

    def record_feedback(self, user_id: int, movie_id: int, rating: float, predicted: float) -> None:
        self.m_feedback.inc()
        with self._lock:
            self.totals["feedback"] += 1
            self._feedback_err.append(float(predicted) - float(rating))
        self._append({"ts": self._now(), "event": "feedback", "model_version": self.model_version,
                      "user": self.pseudonymise(user_id), "movie_id": int(movie_id),
                      "rating": float(rating), "predicted": round(float(predicted), 4)})

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        with self._lock:
            reqs = list(self._requests)
            bins = np.concatenate(list(self._pop_bins)) if self._pop_bins else np.empty(0, dtype=np.int64)
            fb = np.asarray(self._feedback_err, dtype=float)
            lat = np.asarray(self._latencies, dtype=float)
            totals = dict(self.totals)
            n_distinct_items = len(self._recommended_items)

        n = len(reqs)
        act_counts = Counter(r["activity"] for r in reqs)
        actual_act = [act_counts.get(lbl, 0) for lbl in self.activity_labels]
        ref_act = self.reference.get("user_activity_distribution")
        input_psi = psi(ref_act, actual_act) if ref_act and n else float("nan")

        actual_pop = np.bincount(bins, minlength=self.n_bins)[: self.n_bins] if len(bins) else np.zeros(self.n_bins)
        ref_pop = self.reference.get("recommended_popularity_distribution")
        svd_reqs = sum(1 for r in reqs if r["strategy"] == "svd")
        pred_psi = psi(ref_pop, actual_pop) if ref_pop and len(bins) else float("nan")
        for name, val in (("input_user_activity", input_psi), ("prediction_item_popularity", pred_psi)):
            if not math.isnan(val):
                self.m_psi.labels(signal=name).set(val)

        perf: dict[str, Any]
        if len(fb) >= self.min_feedback:
            perf = {"status": "available", "n_labelled": int(len(fb)),
                    "rmse": float(np.sqrt(np.mean(fb ** 2))), "mae": float(np.mean(np.abs(fb))),
                    "mean_error": float(np.mean(fb)),
                    "note": "Computed from delayed feedback ratings (ground truth)."}
        else:
            perf = {"status": "insufficient_labels", "n_labelled": int(len(fb)),
                    "required": self.min_feedback,
                    "note": "No performance claim is made without enough labelled feedback."}

        def rate(key: str) -> float | None:
            denom = totals.get("recommendation_requests", 0)
            return round(totals.get(key, 0) / denom, 4) if denom else None

        return {
            "model_version": self.model_version,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "window_size": self.window,
            "requests_in_window": n,
            "totals": totals,
            "latency_ms": {
                "p50": round(float(np.percentile(lat, 50)) * 1000, 3) if len(lat) else None,
                "p95": round(float(np.percentile(lat, 95)) * 1000, 3) if len(lat) else None,
                "p99": round(float(np.percentile(lat, 99)) * 1000, 3) if len(lat) else None,
            },
            "rates": {"cold_start_fallback": rate("cold_start_fallback"),
                      "unknown_user": rate("unknown_user"),
                      "short_response": rate("short_responses")},
            "recommendation_distribution": {
                "distinct_items_recommended": n_distinct_items,
                "popularity_decile_counts": [int(x) for x in actual_pop],
                "reference_popularity_distribution": ref_pop,
            },
            "input_drift": {
                "signal": "requesting-user activity profile vs training users (PSI)",
                "psi": None if math.isnan(input_psi) else round(input_psi, 4),
                "status": drift_status(input_psi, n, self.min_requests, self.psi_warn, self.psi_alert),
                "observed_counts": dict(zip(self.activity_labels, actual_act)),
            },
            "prediction_drift": {
                "signal": "popularity deciles of recommended items vs build-time reference (PSI)",
                "psi": None if math.isnan(pred_psi) else round(pred_psi, 4),
                "status": drift_status(pred_psi, svd_reqs, self.min_requests, self.psi_warn, self.psi_alert),
                "based_on_svd_requests": svd_reqs,
            },
            "observed_performance": perf,
            "data_quality": {
                "client_errors": totals.get("client_errors", 0),
                "validation_errors": totals.get("error:validation", 0),
                "unknown_movie_ids_in_requests": totals.get("unknown_movie_ids_in_requests", 0),
            },
            "thresholds": {"psi_warning": self.psi_warn, "psi_alert": self.psi_alert,
                           "min_requests_for_drift": self.min_requests,
                           "min_feedback_for_performance": self.min_feedback},
        }


def report_from_log(log_path: Path, reference: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Offline monitoring report rebuilt from an ``events.jsonl`` file."""
    mon = RecommendationMonitor(reference, dict(cfg, window_size=10 ** 7), model_version="offline-report")
    n_bad = 0
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if ev.get("event") == "recommendation":
                with mon._lock:
                    mon.totals["recommendation_requests"] += 1
                    mon.totals["items_recommended"] += ev.get("n_returned", 0)
                    if ev.get("strategy") == "popularity_fallback":
                        mon.totals["cold_start_fallback"] += 1
                    if ev.get("activity") == "unknown_user":
                        mon.totals["unknown_user"] += 1
                    mon._requests.append({"activity": ev.get("activity"), "strategy": ev.get("strategy")})
                    if ev.get("strategy") == "svd":
                        mon._pop_bins.append(np.asarray(ev.get("popularity_bins", []), dtype=np.int64))
                    mon._latencies.append(float(ev.get("latency_ms", 0)) / 1000)
            elif ev.get("event") == "feedback":
                with mon._lock:
                    mon.totals["feedback"] += 1
                    mon._feedback_err.append(float(ev["predicted"]) - float(ev["rating"]))
    out = mon.summary()
    out["source_log"] = str(log_path)
    out["malformed_lines"] = n_bad
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: python src/monitoring.py --log monitoring/events.jsonl [--out report.json]"""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from registry import ModelRegistry
    from settings import get, load_config, resolve_path

    ap = argparse.ArgumentParser(description="Offline monitoring report from the API event log")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", help="write JSON report here")
    ap.add_argument("--version", help="model version whose reference profile to use (default: approved)")
    args = ap.parse_args(argv)
    cfg = load_config()
    reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "models")))
    vdir = reg.version_dir(args.version) if args.version else reg.approved_dir()
    reference = json.loads((vdir / "reference_profile.json").read_text(encoding="utf-8"))
    rep = report_from_log(Path(args.log), reference, cfg.get("monitoring", {}))
    text = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
