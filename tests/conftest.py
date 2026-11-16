"""Shared fixtures: small SYNTHETIC MovieLens-shaped data and isolated config.

Nothing here touches the real dataset, the real model registry, outputs/ or
mlflow.db: every path points into pytest's tmp_path.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SVD_API_NO_AUTOCREATE", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from synthetic_data import make_movies, make_ratings  # noqa: E402,F401

from settings import load_config  # noqa: E402


@pytest.fixture
def ratings() -> pd.DataFrame:
    return make_ratings()


@pytest.fixture
def movies(ratings) -> pd.DataFrame:
    return make_movies(ratings)


@pytest.fixture
def cfg(tmp_path) -> dict:
    """Project config redirected into tmp_path, small and fast."""
    base = copy.deepcopy(load_config(ROOT / "config.yaml", apply_env=False))
    base["model"]["k_values"] = [2, 3, 5, 8]
    base["sgd_mf"]["epochs"] = 3
    base["ranking"]["min_item_ratings"] = 2
    base["registry"]["dir"] = str(tmp_path / "models")
    base["output"]["corrected_dir"] = str(tmp_path / "out")
    base["output"]["plots_dir"] = str(tmp_path / "out" / "plots")
    base["monitoring"]["log_dir"] = str(tmp_path / "monitoring")
    base["monitoring"]["min_requests_for_drift"] = 20
    base["monitoring"]["min_feedback_for_performance"] = 10
    base["retraining"]["report_dir"] = str(tmp_path / "reports")
    base["retraining"]["min_new_ratings"] = 50
    base["mlflow"]["enabled"] = False
    base["dataset"]["movies_path"] = str(tmp_path / "no_movies.csv")
    return base


@pytest.fixture
def trained(cfg, ratings, movies):
    """Run the corrected pipeline once on synthetic data and approve the model."""
    from main import run_pipeline
    summary = run_pipeline(cfg, ratings=ratings, movies=movies, make_plots=False, register=True,
                           approve=True, approval_reason="test fixture")
    return summary
