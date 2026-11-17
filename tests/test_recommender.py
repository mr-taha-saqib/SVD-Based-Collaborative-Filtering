"""Serving artifact: build, reload, recommend, cold start, exclusions."""

from __future__ import annotations

import numpy as np
import pytest

from recommender import ArtifactError, Recommender
from registry import ModelRegistry


@pytest.fixture
def rec(trained, cfg):
    return Recommender.load(ModelRegistry(cfg["registry"]["dir"]).approved_dir())


def test_reload_reproduces_predictions(trained, cfg, ratings):
    from data_loader import train_val_test_split
    from training import decompose
    sp = train_val_test_split(ratings, 0.1, 0.2, 42)
    dec = decompose(sp.train, 8)
    model = dec.model(trained["selection"]["selected_k"])
    rec = Recommender.load(ModelRegistry(cfg["registry"]["dir"]).approved_dir())
    for uid, mid in sp.test[["userId", "movieId"]].head(20).itertuples(index=False):
        if mid in rec.item_index:
            exp = model.predict(dec.user_idx[uid], dec.item_idx[mid])
            assert rec.predict_rating(uid, mid)[0] == pytest.approx(exp, abs=1e-10)


def test_recommendations_exclude_rated_and_are_sorted(rec, ratings):
    uid = int(ratings.userId.iloc[0])
    res = rec.recommend(uid, n=10)
    assert res.strategy == "svd" and res.known_user
    ids = [i.movie_id for i in res.items]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert not set(ids) & set(ratings.loc[ratings.userId == uid, "movieId"])
    scores = [i.score for i in res.items]
    assert scores == sorted(scores, reverse=True)
    assert all(0.5 <= i.predicted_rating <= 5.0 for i in res.items)
    assert [i.rank for i in res.items] == list(range(1, 11))


def test_rated_items_can_be_included_on_request(rec, ratings):
    uid = int(ratings.userId.iloc[0])
    allowed = rec.recommend(uid, n=200, exclude_rated=False)
    assert set(i.movie_id for i in allowed.items) & set(ratings.loc[ratings.userId == uid, "movieId"])


def test_unknown_user_gets_popularity_fallback(rec, ratings):
    res = rec.recommend(10 ** 9, n=5)
    assert res.strategy == "popularity_fallback" and not res.known_user
    counts = ratings.groupby("movieId").size()
    top = [i.movie_id for i in res.items]
    assert len(top) == 5
    assert all(i.predicted_rating is None for i in res.items)
    # most-popular order (counts non-increasing)
    assert [rec.item_counts[rec.item_index[m]] for m in top] == sorted(
        [rec.item_counts[rec.item_index[m]] for m in top], reverse=True)
    assert counts.loc[top[0]] >= counts.median()


def test_extra_exclusions_and_unknown_ids(rec, ratings):
    uid = int(ratings.userId.iloc[0])
    first = rec.recommend(uid, n=3).items[0].movie_id
    res = rec.recommend(uid, n=3, exclude_movie_ids=[first, 987654321])
    assert first not in [i.movie_id for i in res.items]
    assert res.unknown_excluded_movie_ids == [987654321]


def test_titles_come_from_movies_file(rec):
    item = rec.recommend(1, n=1).items[0]
    assert item.title == f"Synthetic Movie {item.movie_id} (2000)"
    assert item.genres in (["Drama", "Comedy"], [])


def test_predict_fallback_methods(rec):
    u, m = int(rec.user_ids[0]), int(rec.item_ids[0])
    assert rec.predict_rating(u, m)[1] == "svd"
    assert rec.predict_rating(u, 10 ** 9)[1] == "user_bias_only"
    assert rec.predict_rating(10 ** 9, m)[1] == "item_bias_only"
    pred, method = rec.predict_rating(10 ** 9, 10 ** 9)
    assert method == "global_mean" and pred == pytest.approx(np.clip(rec.model.mu, 0.5, 5.0))


def test_invalid_n_raises(rec):
    with pytest.raises(ValueError):
        rec.recommend(1, n=0)


def test_tampered_artifact_is_rejected(trained, cfg):
    d = ModelRegistry(cfg["registry"]["dir"]).approved_dir()
    data = bytearray((d / "model.npz").read_bytes())
    data[-10] ^= 0xFF
    (d / "model.npz").write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="Checksum"):
        Recommender.load(d)


def test_missing_artifact(tmp_path):
    with pytest.raises(ArtifactError):
        Recommender.load(tmp_path)
