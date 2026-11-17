"""Retraining: data gates, promotion gate, promotion/rejection, rollback, dry-run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_synthetic_feedback import generate  # noqa: E402

import retrain as rt  # noqa: E402
from recommender import Recommender  # noqa: E402
from registry import ModelRegistry  # noqa: E402

RULES = {"min_rmse_improvement": 0.0, "max_base_rmse_degradation": 0.002, "max_ndcg_drop": 0.01}
APPROVED = {"rmse": 0.90, "base_rmse": 0.90, "ndcg@10": 0.10}


class TestPromotionGate:
    def test_passes_when_all_criteria_met(self):
        g = rt.promotion_gate({"rmse": 0.89, "base_rmse": 0.901, "ndcg@10": 0.095}, APPROVED, RULES, "ndcg@10")
        assert g.passed and len(g.criteria) == 4

    @pytest.mark.parametrize("cand,failed", [
        ({"rmse": 0.91, "base_rmse": 0.90, "ndcg@10": 0.1}, "validation_rmse_improves"),
        ({"rmse": 0.89, "base_rmse": 0.95, "ndcg@10": 0.1}, "base_validation_rmse_not_degraded"),
        ({"rmse": 0.89, "base_rmse": 0.90, "ndcg@10": 0.05}, "ndcg_not_dropped"),
        ({"rmse": float("nan"), "base_rmse": 0.90, "ndcg@10": 0.1}, "candidate_metrics_finite"),
    ])
    def test_each_criterion_can_block(self, cand, failed):
        g = rt.promotion_gate(cand, APPROVED, RULES, "ndcg@10")
        assert not g.passed
        assert failed in [c["criterion"] for c in g.criteria if not c["passed"]]

    def test_min_improvement_is_enforced(self):
        rules = dict(RULES, min_rmse_improvement=0.02)
        g = rt.promotion_gate({"rmse": 0.89, "base_rmse": 0.9, "ndcg@10": 0.1}, APPROVED, rules, "ndcg@10")
        assert not g.passed


def _approved(cfg):
    reg = ModelRegistry(cfg["registry"]["dir"])
    return reg, Recommender.load(reg.approved_dir())


def test_feedback_validation_drops_known_pairs_and_invalid_rows(trained, cfg, ratings):
    reg, rec = _approved(cfg)
    fb = generate(rec, 200, "invalid", seed=1)
    fb = pd.concat([fb, ratings.head(7)], ignore_index=True)       # pairs already in base data
    tr, va, rep = rt.prepare_feedback(fb, ratings, cfg, set(rec.user_index), set(rec.item_index))
    assert rep["dropped_existing_pairs"] >= 7
    assert rep["validation"]["n_out_of_range"] >= 1 and rep["validation"]["n_non_integer_ids"] >= 2
    assert rep["requirements_passed"]
    assert len(tr) + len(va) == rep["n_new_ratings"] and len(va) > 0
    assert not set(zip(tr.userId, tr.movieId)) & set(zip(va.userId, va.movieId))


def test_too_little_feedback_is_rejected_without_training(trained, cfg, ratings):
    reg, rec = _approved(cfg)
    before = reg.approved_version()
    rep = rt.run_retraining(cfg, generate(rec, 10, "consistent"), base_ratings=ratings)
    assert rep["decision"] == "rejected_data_requirements" and not rep["promoted"]
    assert reg.approved_version() == before and len(reg.list_versions()) == 1


def test_new_user_feedback_is_promoted_and_previous_preserved(trained, cfg, ratings, movies):
    reg, rec = _approved(cfg)
    before = reg.approved_version()
    rep = rt.run_retraining(cfg, generate(rec, 600, "new_users", seed=3), base_ratings=ratings, movies=movies)
    assert rep["decision"] in ("promoted", "rejected_by_gate")
    assert rep["test_partition_used"] is False
    if rep["promoted"]:
        assert reg.approved_version() == rep["candidate_version"] != before
        assert reg.exists(before)                                   # previous model kept
        assert reg.rollback("test") == before
    else:
        assert reg.approved_version() == before
    assert Path(cfg["retraining"]["report_dir"]).glob("retraining-*.md")


def test_noise_feedback_never_changes_approved_when_gate_fails(trained, cfg, ratings, movies):
    reg, rec = _approved(cfg)
    before = reg.approved_version()
    rep = rt.run_retraining(cfg, generate(rec, 600, "noise", seed=5), base_ratings=ratings, movies=movies)
    if not rep["gate"]["passed"]:
        assert reg.approved_version() == before
        assert reg.metadata(rep["candidate_version"])["status"] == "rejected"


def test_dry_run_changes_nothing(trained, cfg, ratings):
    reg, rec = _approved(cfg)
    state = (reg.approved_version(), reg.list_versions())
    rep = rt.run_retraining(cfg, generate(rec, 600, "new_users", seed=4), dry_run=True, base_ratings=ratings)
    assert rep["decision"].startswith("dry_run")
    assert (reg.approved_version(), reg.list_versions()) == state


def test_retraining_never_reads_the_test_partition(trained, cfg, ratings, monkeypatch):
    from data_loader import train_val_test_split
    sp = train_val_test_split(ratings, cfg["split"]["validation_fraction"], cfg["split"]["test_fraction"],
                              cfg["split"]["random_state"])
    test_pairs = set(zip(sp.test.userId, sp.test.movieId))
    seen_frames = []
    real_eval, real_dec = rt.evaluate_pairs, rt.decompose
    monkeypatch.setattr(rt, "evaluate_pairs", lambda n, m, df, u, i: (seen_frames.append(df), real_eval(n, m, df, u, i))[1])
    monkeypatch.setattr(rt, "decompose", lambda df, *a, **k: (seen_frames.append(df), real_dec(df, *a, **k))[1])
    _, rec = _approved(cfg)
    rt.run_retraining(cfg, generate(rec, 600, "new_users", seed=6), dry_run=True, base_ratings=ratings)
    assert seen_frames
    for df in seen_frames:
        assert not set(zip(df.userId, df.movieId)) & test_pairs


def test_rejects_mismatched_base_dataset(trained, cfg, ratings):
    _, rec = _approved(cfg)
    other = ratings.copy()
    other.loc[0, "rating"] = 0.5 if other.loc[0, "rating"] != 0.5 else 1.0
    with pytest.raises(rt.RetrainingError, match="fingerprint"):
        rt.run_retraining(cfg, generate(rec, 600, "consistent"), base_ratings=other)


def test_no_approved_model(cfg, ratings):
    with pytest.raises(rt.RetrainingError, match="No approved model"):
        rt.run_retraining(cfg, ratings.head(10), base_ratings=ratings)


def test_cli_rollback_without_history_fails_cleanly(trained, cfg, monkeypatch, tmp_path):
    from registry import RegistryError
    with pytest.raises(RegistryError):
        ModelRegistry(cfg["registry"]["dir"]).rollback()
    assert np.isfinite(1.0)


def test_back_to_back_runs_keep_separate_reports(trained, cfg, ratings):
    _, rec = _approved(cfg)
    for _ in range(3):
        rt.run_retraining(cfg, generate(rec, 10, "consistent"), base_ratings=ratings)
    assert len(list(Path(cfg["retraining"]["report_dir"]).glob("retraining-*.md"))) == 3


def test_rollback_cli_without_history_exits_cleanly(trained, cfg, monkeypatch):
    monkeypatch.setenv("SVD_MODEL_DIR", cfg["registry"]["dir"])
    monkeypatch.setenv("SVD_MLFLOW_ENABLED", "false")
    assert rt.main(["--rollback", "--reason", "test"]) == 1
