"""MovieTweetings loader: fixed preparation, checksum pinning, privacy (users.dat never read).

All inputs are tiny SYNTHETIC files in tmp_path; no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import movietweetings as mt
from data_loader import validate_ratings
from run_movietweetings import build_config


def _write_dataset(d, n_users=30, n_items=25, seed=0, with_zero=True):
    rng = np.random.default_rng(seed)
    lines = []
    for u in range(1, n_users + 1):
        for m in range(1, n_items + 1):
            if rng.random() < 0.9:
                lines.append(f"{u}::{1000 + m:07d}::{int(rng.integers(1, 11))}::{1600000000 + u * 100 + m}")
    lines.append(f"{n_users + 1}::{1001:07d}::7::1600000000")          # sparse user, removed by the core
    if with_zero:
        lines.append(f"1::{1002:07d}::0::1700000000")                  # out-of-scale 0 rating
    (d / "ratings.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    movies = [f"{1000 + m:07d}::Movie {m} (2020)::Drama|Comedy" for m in range(1, n_items + 1)]
    movies.append(f"{1000 + n_items + 1:07d}::No Genre (2021)::")
    (d / "movies.dat").write_text("\n".join(movies) + "\n", encoding="utf-8")
    (d / "LICENSE").write_text("MIT\n", encoding="utf-8")


def test_prepare_rescales_drops_zero_and_applies_core(tmp_path):
    _write_dataset(tmp_path)
    raw = mt.read_ratings(tmp_path / "ratings.dat")
    df, rep = mt.prepare(raw, min_user=20, min_item=20)
    assert rep.dropped_zero_ratings >= 1 and (df["rating"] > 0).all()
    assert df["rating"].between(0.5, 5.0).all()
    assert np.allclose((df["rating"] * 2) % 1, 0)                       # exactly the 0.5 grid
    assert (df["userId"].value_counts() >= 20).all() and (df["movieId"].value_counts() >= 20).all()
    assert rep.ratings == len(df) and rep.users == df["userId"].nunique()
    validate_ratings(df, rating_scale=(0.5, 5.0))                       # accepted by the shared validator


def test_duplicate_pairs_keep_latest(tmp_path):
    raw = pd.DataFrame({"userId": [1, 1], "movieId": [5, 5], "rating": [2, 8], "timestamp": [1, 2]})
    df, rep = mt.prepare(raw, min_user=1, min_item=1)
    assert rep.dropped_duplicate_pairs == 1 and df["rating"].tolist() == [4.0]


def test_out_of_range_ratings_raise():
    raw = pd.DataFrame({"userId": [1], "movieId": [5], "rating": [11], "timestamp": [1]})
    with pytest.raises(ValueError):
        mt.prepare(raw, min_user=1, min_item=1)


def test_movies_parsing_and_missing_genres(tmp_path):
    _write_dataset(tmp_path)
    movies = mt.read_movies(tmp_path / "movies.dat")
    assert {"movieId", "title", "genres"} <= set(movies.columns)
    assert movies.loc[movies["title"] == "No Genre (2021)", "genres"].item() == "(no genres listed)"


def test_load_never_reads_users_file_and_records_checksums(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    (tmp_path / "users.dat").write_text("1::123456789\n", encoding="utf-8")
    opened = []
    real_open = open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", spy)
    ratings, movies, rep = mt.load_movietweetings(tmp_path, download=False)
    assert not any(p.endswith("users.dat") for p in opened)
    assert set(rep.file_sha256) == {"ratings.dat", "movies.dat", "LICENSE"}
    assert set(movies["movieId"]) <= set(ratings["movieId"])


def test_checksum_mismatch_aborts(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(mt, "download_archive", lambda *a, **k: None)   # files already "downloaded"
    with pytest.raises(ValueError, match="pinned checksum"):
        mt.download_movietweetings(tmp_path)


def test_separate_locations_do_not_touch_movielens(cfg):
    out = build_config(cfg)
    assert out["registry"]["dir"] != cfg["registry"]["dir"]
    assert out["output"]["corrected_dir"] == "outputs/external/movietweetings"
    assert out["mlflow"]["experiment"] != cfg["mlflow"]["experiment"]
    assert out["sgd_mf"]["enabled"] is False
    assert cfg["dataset"]["name"] != out["dataset"]["name"]              # input config unchanged
