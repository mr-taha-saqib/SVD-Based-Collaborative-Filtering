"""
recommender.py
==============
Serving artifact for the SVD recommender: build, save, load and query.

Artifact directory (one per registry version)::

    model.npz              numeric arrays only (loaded with allow_pickle=False)
    metadata.json          version, config, metrics, data fingerprints, environment
    movies.csv             movieId,title,genres for items in the model vocabulary
                           (from MovieLens movies.csv; absent if not supplied)
    reference_profile.json monitoring reference distributions

ID mapping: the model works with matrix indices. ``user_ids[r]`` is the MovieLens
``userId`` of row ``r`` and ``item_ids[c]`` the ``movieId`` of column ``c``. Titles
and genres are joined on ``movieId`` from the dataset's own ``movies.csv``; no
metadata is invented. Items without metadata are returned with ``title = null``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ranking import top_n
from svd_model import SVDModel

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT = "svd-recommender/v1"
N_POPULARITY_BINS = 10
ACTIVITY_EDGES = (0, 20, 50, 100, 200, 500, np.inf)   # training ratings per user


class ArtifactError(RuntimeError):
    """Raised when a model artifact is missing or inconsistent."""


def _csr_from_pairs(rows: np.ndarray, cols: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr), cols.astype(np.int64)


def popularity_bins(item_counts: np.ndarray, n_bins: int = N_POPULARITY_BINS) -> np.ndarray:
    """Assign each item a popularity decile (0 = least, n_bins-1 = most popular)."""
    ranks = pd.Series(item_counts).rank(method="first").to_numpy()
    return np.minimum((ranks - 1) * n_bins // len(item_counts), n_bins - 1).astype(np.int64)


def activity_bucket(n_ratings: int) -> str:
    for lo, hi in zip(ACTIVITY_EDGES[:-1], ACTIVITY_EDGES[1:]):
        if lo <= n_ratings < hi:
            return f"{lo}-{'inf' if hi == np.inf else int(hi) - 1}"
    return "unknown"


def activity_bucket_labels() -> list[str]:
    return [activity_bucket(int(lo)) for lo in ACTIVITY_EDGES[:-1]] + ["unknown_user"]


def build_artifact(
    out_dir: str | Path,
    model: SVDModel,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    train_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    movies: pd.DataFrame | None,
    metadata: dict[str, Any],
    min_item_ratings: int = 5,
    reference_top_k: int = 10,
) -> str:
    """Write a complete serving artifact. Returns its content SHA-256.

    Args:
        train_df: ratings the model was fitted on (for counts / popularity).
        seen_df: all interactions known at build time; their items are excluded
            from that user's recommendations. Used for serving only, never for
            fitting or evaluation.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    item_ids = np.asarray(item_ids, dtype=np.int64)
    u_index = {int(u): r for r, u in enumerate(user_ids)}
    i_index = {int(m): c for c, m in enumerate(item_ids)}

    item_counts = np.zeros(len(item_ids), dtype=np.int64)
    tr_cols = train_df["movieId"].map(i_index).dropna().astype(np.int64).to_numpy()
    np.add.at(item_counts, tr_cols, 1)
    user_counts = np.zeros(len(user_ids), dtype=np.int64)
    tr_rows = train_df["userId"].map(u_index).dropna().astype(np.int64).to_numpy()
    np.add.at(user_counts, tr_rows, 1)

    s_rows = seen_df["userId"].map(u_index)
    s_cols = seen_df["movieId"].map(i_index)
    ok = s_rows.notna() & s_cols.notna()
    indptr, indices = _csr_from_pairs(s_rows[ok].astype(np.int64).to_numpy(),
                                      s_cols[ok].astype(np.int64).to_numpy(), len(user_ids))

    arrays = model.to_arrays()
    arrays.update({
        "user_ids": user_ids, "item_ids": item_ids,
        "item_train_counts": item_counts, "user_train_counts": user_counts,
        "seen_indptr": indptr, "seen_indices": indices,
    })
    np.savez_compressed(out / "model.npz", **arrays)

    if movies is not None and len(movies):
        movies[movies["movieId"].isin(set(item_ids.tolist()))].to_csv(
            out / "movies.csv", index=False, lineterminator="\n")

    # Monitoring reference: popularity deciles of recommended items for known users,
    # and the activity profile of training users.
    rec = Recommender(model=model, user_ids=user_ids, item_ids=item_ids,
                      item_counts=item_counts, user_counts=user_counts,
                      seen_indptr=indptr, seen_indices=indices,
                      metadata={"serving_policy": {"min_item_ratings": min_item_ratings}},
                      movies=None)
    bins = popularity_bins(item_counts)
    hist = np.zeros(N_POPULARITY_BINS, dtype=float)
    sample_rows = np.arange(len(user_ids))
    for r in sample_rows:
        idx = rec._top_for_row(int(r), reference_top_k, exclude_rated=True, extra=())
        np.add.at(hist, bins[idx], 1)
    act_labels = activity_bucket_labels()
    act = pd.Series([activity_bucket(int(c)) for c in user_counts]).value_counts()
    reference = {
        "popularity_bin_of_item_index": "item_pop_bins in reference_profile.json",
        "item_pop_bins": bins.tolist(),
        "recommended_popularity_distribution": (hist / max(hist.sum(), 1)).tolist(),
        "user_activity_labels": act_labels,
        "user_activity_distribution": [float(act.get(lbl, 0)) / len(user_counts) for lbl in act_labels],
        "reference_top_k": int(reference_top_k),
        "n_reference_users": int(len(sample_rows)),
    }
    (out / "reference_profile.json").write_text(json.dumps(reference), encoding="utf-8")

    content_hash = hashlib.sha256((out / "model.npz").read_bytes()).hexdigest()
    meta = dict(metadata)
    meta.update({
        "artifact_format": ARTIFACT_FORMAT,
        "content_sha256": content_hash,
        "n_users": int(len(user_ids)), "n_items": int(len(item_ids)),
        "serving_policy": {"min_item_ratings": int(min_item_ratings),
                           "exclude_rated": True,
                           "cold_start": "most-popular items by training rating count"},
        "has_movie_metadata": bool(movies is not None and len(movies)),
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return content_hash


@dataclass
class Recommendation:
    movie_id: int
    rank: int
    score: float
    predicted_rating: float | None
    title: str | None = None
    genres: list[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    user_id: int
    strategy: str              # "svd" | "popularity_fallback"
    known_user: bool
    items: list[Recommendation]
    unknown_excluded_movie_ids: list[int] = field(default_factory=list)


class Recommender:
    """Loaded, query-able recommender. Thread-safe for reads (immutable arrays)."""

    def __init__(self, model: SVDModel, user_ids: np.ndarray, item_ids: np.ndarray,
                 item_counts: np.ndarray, user_counts: np.ndarray,
                 seen_indptr: np.ndarray, seen_indices: np.ndarray,
                 metadata: dict[str, Any], movies: pd.DataFrame | None,
                 reference: dict[str, Any] | None = None):
        self.model = model
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.item_counts = item_counts
        self.user_counts = user_counts
        self.seen_indptr = seen_indptr
        self.seen_indices = seen_indices
        self.metadata = metadata
        self.reference = reference or {}
        self.user_index = {int(u): r for r, u in enumerate(user_ids)}
        self.item_index = {int(m): c for c, m in enumerate(item_ids)}
        min_r = int(metadata.get("serving_policy", {}).get("min_item_ratings", 1))
        self.candidate_mask = item_counts >= min_r
        clip = metadata.get("config", {}).get("model", {}).get("rating_clip", [0.5, 5.0])
        self.clip = (float(clip[0]), float(clip[1]))
        self._popular_order = np.lexsort((np.arange(len(item_counts)), -item_counts))
        self._titles: dict[int, tuple[str, list[str]]] = {}
        if movies is not None and len(movies):
            for row in movies.itertuples(index=False):
                genres = [] if pd.isna(row.genres) or row.genres == "(no genres listed)" \
                    else str(row.genres).split("|")
                self._titles[int(row.movieId)] = (str(row.title), genres)

    # ── loading ───────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, artifact_dir: str | Path) -> Recommender:
        d = Path(artifact_dir)
        meta_path, npz_path = d / "metadata.json", d / "model.npz"
        if not meta_path.exists() or not npz_path.exists():
            raise ArtifactError(f"Incomplete model artifact at {d}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("artifact_format") != ARTIFACT_FORMAT:
            raise ArtifactError(f"Unsupported artifact format {metadata.get('artifact_format')!r}")
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if metadata.get("content_sha256") and digest != metadata["content_sha256"]:
            raise ArtifactError(f"Checksum mismatch for {npz_path}: artifact is corrupted")
        with np.load(npz_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        model = SVDModel.from_arrays(arrays)
        movies = pd.read_csv(d / "movies.csv") if (d / "movies.csv").exists() else None
        ref_path = d / "reference_profile.json"
        reference = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
        rec = cls(model=model, user_ids=arrays["user_ids"], item_ids=arrays["item_ids"],
                  item_counts=arrays["item_train_counts"], user_counts=arrays["user_train_counts"],
                  seen_indptr=arrays["seen_indptr"], seen_indices=arrays["seen_indices"],
                  metadata=metadata, movies=movies, reference=reference)
        logger.info("Loaded model %s (k=%d, %d users, %d items)",
                    metadata.get("version_id"), model.k, len(rec.user_ids), len(rec.item_ids))
        return rec

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def version(self) -> str:
        return str(self.metadata.get("version_id", "unversioned"))

    def is_known_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_index

    def is_known_item(self, movie_id: int) -> bool:
        return int(movie_id) in self.item_index

    def rated_items(self, user_id: int) -> list[int]:
        r = self.user_index.get(int(user_id))
        if r is None:
            return []
        cols = self.seen_indices[self.seen_indptr[r]:self.seen_indptr[r + 1]]
        return [int(self.item_ids[c]) for c in cols]

    def user_activity(self, user_id: int) -> str:
        r = self.user_index.get(int(user_id))
        return "unknown_user" if r is None else activity_bucket(int(self.user_counts[r]))

    # ── core ranking ──────────────────────────────────────────────────────────
    def _top_for_row(self, row: int, n: int, exclude_rated: bool, extra: Iterable[int]) -> np.ndarray:
        scores = self.model.score_users(np.array([row]))[0]
        allowed = self.candidate_mask.copy()
        if exclude_rated:
            allowed[self.seen_indices[self.seen_indptr[row]:self.seen_indptr[row + 1]]] = False
        extra_cols = [c for c in extra]
        if extra_cols:
            allowed[np.asarray(extra_cols, dtype=np.int64)] = False
        return top_n(scores, allowed, n)

    def _describe(self, movie_id: int) -> tuple[str | None, list[str]]:
        return self._titles.get(int(movie_id), (None, []))

    def recommend(self, user_id: int, n: int = 10, exclude_rated: bool = True,
                  exclude_movie_ids: Iterable[int] = ()) -> RecommendationResult:
        """Top-``n`` recommendations. Unknown users get the popularity fallback."""
        if n <= 0:
            raise ValueError("n must be positive")
        excl = list(dict.fromkeys(int(m) for m in exclude_movie_ids))
        unknown_excl = [m for m in excl if m not in self.item_index]
        extra_cols = [self.item_index[m] for m in excl if m in self.item_index]
        row = self.user_index.get(int(user_id))

        if row is not None:
            cols = self._top_for_row(row, n, exclude_rated, extra_cols)
            scores = self.model.score_users(np.array([row]))[0][cols]
            strategy = "svd"
        else:
            allowed = self.candidate_mask.copy()
            if extra_cols:
                allowed[np.asarray(extra_cols)] = False
            order = [c for c in self._popular_order if allowed[c]][:n]
            cols = np.asarray(order, dtype=np.int64)
            scores = self.item_counts[cols].astype(float)
            strategy = "popularity_fallback"

        items = []
        for rank, (c, sc) in enumerate(zip(cols.tolist(), scores.tolist()), start=1):
            mid = int(self.item_ids[c])
            title, genres = self._describe(mid)
            pred = float(np.clip(sc, *self.clip)) if strategy == "svd" else None
            items.append(Recommendation(movie_id=mid, rank=rank, score=round(float(sc), 6),
                                        predicted_rating=None if pred is None else round(pred, 4),
                                        title=title, genres=genres))
        return RecommendationResult(user_id=int(user_id), strategy=strategy,
                                    known_user=row is not None, items=items,
                                    unknown_excluded_movie_ids=unknown_excl)

    def predict_rating(self, user_id: int, movie_id: int) -> tuple[float, str]:
        """Predicted rating and the method used ('svd', 'item_fallback', ...)."""
        u = self.user_index.get(int(user_id), -1)
        i = self.item_index.get(int(movie_id), -1)
        if u >= 0 and i >= 0:
            return float(self.model.predict(u, i, clip=self.clip)), "svd"
        pred = float(self.model.fallback_batch(np.array([u]), np.array([i]), clip=self.clip)[0])
        method = ("user_bias_only" if u >= 0 else "item_bias_only" if i >= 0 else "global_mean")
        return pred, method
