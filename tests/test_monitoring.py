"""Monitoring: PSI, drift status, delayed-label performance, pseudonymised logs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from monitoring import RecommendationMonitor, drift_status, psi, report_from_log
from recommender import Recommender
from registry import ModelRegistry


def test_psi_basics():
    assert psi([0.5, 0.5], [50, 50]) == pytest.approx(0.0, abs=1e-12)
    assert psi([0.5, 0.5], [90, 10]) > 0.25
    assert np.isnan(psi([0.5, 0.5], [0, 0]))


def test_drift_status_thresholds():
    assert drift_status(0.05, 100, 50, 0.1, 0.25) == "ok"
    assert drift_status(0.15, 100, 50, 0.1, 0.25) == "warning"
    assert drift_status(0.3, 100, 50, 0.1, 0.25) == "alert"
    assert drift_status(0.3, 10, 50, 0.1, 0.25) == "insufficient_data"


@pytest.fixture
def rec(trained, cfg):
    return Recommender.load(ModelRegistry(cfg["registry"]["dir"]).approved_dir())


def _serve(mon, rec, user_ids, n=5):
    for u in user_ids:
        res = rec.recommend(int(u), n=n)
        cols = [rec.item_index[i.movie_id] for i in res.items]
        mon.record_recommendation(int(u), rec.user_activity(int(u)), res.strategy, cols, n, 0, 0.001)


def test_normal_traffic_is_not_flagged_but_controlled_drift_is(rec, cfg, tmp_path):
    rng = np.random.default_rng(0)
    normal = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version)
    _serve(normal, rec, rng.choice(rec.user_ids, 300))
    s = normal.summary()
    assert s["input_drift"]["status"] in ("ok", "warning")
    assert s["prediction_drift"]["status"] in ("ok", "warning")

    drifted = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version)
    _serve(drifted, rec, range(10 ** 6, 10 ** 6 + 300))      # all unknown users
    d = drifted.summary()
    assert d["input_drift"]["status"] == "alert"
    assert d["rates"]["cold_start_fallback"] == 1.0
    assert d["prediction_drift"]["status"] == "insufficient_data"   # no SVD responses at all
    assert d["prediction_drift"]["psi"] is None
    assert d["observed_performance"]["status"] == "insufficient_labels"


def test_performance_requires_enough_labels(rec, cfg):
    mon = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version)
    for _ in range(5):
        mon.record_feedback(1, 100, 4.0, 3.0)
    assert mon.summary()["observed_performance"]["status"] == "insufficient_labels"
    for _ in range(10):
        mon.record_feedback(1, 100, 4.0, 3.0)
    perf = mon.summary()["observed_performance"]
    assert perf["status"] == "available" and perf["rmse"] == pytest.approx(1.0)


def test_event_log_is_pseudonymised_and_replayable(rec, cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SVD_MONITORING_SALT", "pepper")
    mon = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version, tmp_path / "mon")
    _serve(mon, rec, rec.user_ids[:40])
    lines = (tmp_path / "mon" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 40
    ev = json.loads(lines[0])
    assert ev["user"] != str(int(rec.user_ids[0])) and len(ev["user"]) == 16
    rep = report_from_log(tmp_path / "mon" / "events.jsonl", rec.reference, cfg["monitoring"])
    assert rep["totals"]["recommendation_requests"] == 40
    assert rep["malformed_lines"] == 0


def test_prometheus_registry_is_per_instance(rec, cfg):
    a = RecommendationMonitor(rec.reference, cfg["monitoring"], "a")
    b = RecommendationMonitor(rec.reference, cfg["monitoring"], "b")
    a.record_error("validation")
    assert a.totals["error:validation"] == 1 and b.totals["error:validation"] == 0


@pytest.mark.parametrize("salt", ["", "change-me-to-a-random-string"])
def test_missing_or_placeholder_salt_is_replaced_by_random_salt(rec, cfg, monkeypatch, salt):
    """An empty or published salt would make pseudonymised small integer ids reversible."""
    import hashlib

    monkeypatch.setenv("SVD_MONITORING_SALT", salt)
    a = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version)
    b = RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version)
    assert a.salt not in ("", salt) and len(a.salt) == 32
    assert a.salt != b.salt
    # the pseudonym cannot be recomputed from the user id with the known salt
    known = hashlib.sha256(f"{salt}:1".encode()).hexdigest()[:16]
    assert a.pseudonymise(1) != known


def test_configured_salt_is_used(rec, cfg, monkeypatch):
    monkeypatch.setenv("SVD_MONITORING_SALT", "pepper")
    assert RecommendationMonitor(rec.reference, cfg["monitoring"], rec.version).salt == "pepper"
