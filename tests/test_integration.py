"""End-to-end on synthetic data: pipeline -> artifacts -> MLflow -> API -> retraining."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import run_pipeline


def test_pipeline_outputs_and_selection(cfg, ratings, movies):
    s = run_pipeline(cfg, ratings=ratings, movies=movies, make_plots=True, register=True, approve=False)
    out = Path(cfg["output"]["corrected_dir"])
    for f in ("final_metrics.json", "k_sweep_validation.csv", "results_validation.csv",
              "results_test.csv", "split_manifest.json", "data_quality.json"):
        assert (out / f).exists(), f
    assert any((out / "plots").glob("*.png"))
    sweep = pd.read_csv(out / "k_sweep_validation.csv")
    assert s["selection"]["selected_k"] == int(sweep.sort_values(["val_rmse", "k"]).iloc[0]["k"])
    assert s["registry"]["approved"] is False                       # never auto-approved
    reg_state = json.loads((Path(cfg["registry"]["dir"]) / "registry.json").read_text())
    assert reg_state["approved"] is None
    # SVD beats the global-mean baseline on held-out data (signal exists in the fixture)
    test = pd.read_csv(out / "results_test.csv").set_index("model")
    assert test.loc[f"SVD k={s['selection']['selected_k']}", "rmse"] < test.loc["Global Mean", "rmse"]


def test_selection_never_sees_test(cfg, ratings, monkeypatch):
    import main as main_mod
    from data_loader import train_val_test_split
    sp = train_val_test_split(ratings, 0.1, 0.2, 42)
    test_pairs = set(zip(sp.test.userId, sp.test.movieId))
    captured = {}
    real = main_mod.select_rank

    def spy(dec, ks, val_df):
        captured["val"] = val_df
        return real(dec, ks, val_df)

    monkeypatch.setattr(main_mod, "select_rank", spy)
    run_pipeline(cfg, ratings=ratings, make_plots=False, register=False)
    assert not set(zip(captured["val"].userId, captured["val"].movieId)) & test_pairs


def test_mlflow_records_runs_and_registry_alias(cfg, ratings, movies, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    cfg["mlflow"].update({"enabled": True, "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
                          "artifact_location": str(tmp_path / "mlartifacts"), "register_models": True})
    s = run_pipeline(cfg, ratings=ratings, movies=movies, make_plots=False, register=True, approve=True)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    run = mlflow.get_run(s["mlflow_run_id"])
    assert run.data.params["selected_k"] == str(s["selection"]["selected_k"])
    assert run.data.params["selection_metric"] == "validation_rmse"
    assert "val_rmse" in run.data.metrics and "test_rmse" in run.data.metrics
    assert run.data.tags["dataset_sha256"] == s["dataset"]["sha256"]
    mv = mlflow.MlflowClient().get_model_version_by_alias(cfg["registry"]["model_name"], "champion")
    assert mv.run_id == s["mlflow_run_id"]


def test_full_lifecycle_with_api(cfg, ratings, movies, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from generate_synthetic_feedback import generate

    from api import create_app
    from recommender import Recommender
    from registry import ModelRegistry
    from retrain import run_retraining

    s = run_pipeline(cfg, ratings=ratings, movies=movies, make_plots=False, approve=True)
    reg = ModelRegistry(cfg["registry"]["dir"])
    app = create_app(cfg, feedback_path=tmp_path / "fb.csv", monitoring_dir=tmp_path / "mon")
    with TestClient(app) as c:
        assert c.get("/health").json()["model_version"] == s["registry"]["version_id"]
        for u in ratings.userId.unique()[:30]:
            c.get(f"/recommendations/{u}")
        rec = Recommender.load(reg.approved_dir())
        fb = generate(rec, 600, "new_users", seed=9)
        rep = run_retraining(cfg, fb, base_ratings=ratings, movies=movies)
        if rep["promoted"]:
            r = c.post("/model/reload")             # disabled without a token
            assert r.status_code == 404
        assert (tmp_path / "mon" / "events.jsonl").exists()
