"""
ranking.py
==========
Top-N recommendation metrics computed on a held-out partition.

Protocol (identical for every method):

* Candidate items: items in the training vocabulary with at least
  ``min_item_ratings`` training ratings (the serving policy, fixed a priori).
* Excluded items: everything the user rated in the partitions the model is
  allowed to know about (train when evaluating on validation; train and
  validation when evaluating on test). They are never recommended.
* Relevant items: held-out ratings >= ``relevance_threshold``. Relevant items
  that are not recommendable (unseen in training) still count in the Recall
  denominator, so the metric is not flattered.
* Users with no relevant held-out item are skipped (reported).

Metrics: Precision@K, Recall@K, NDCG@K (binary relevance, log2 discount),
HitRate@K, catalogue coverage (share of candidate items recommended to at least
one evaluated user).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    method: str
    partition: str
    k: int
    precision: float
    recall: float
    ndcg: float
    hit_rate: float
    catalog_coverage: float
    n_users_evaluated: int
    n_users_skipped: int
    n_relevant_not_recommendable: int

    def to_dict(self) -> dict:
        return asdict(self)


def top_n(scores: np.ndarray, allowed: np.ndarray, n: int) -> np.ndarray:
    """Indices of the ``n`` highest scores among ``allowed`` (descending, stable ties)."""
    masked = np.where(allowed, scores, -np.inf)
    n_allowed = int(allowed.sum())
    n = min(n, n_allowed)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-masked, n - 1)[:n]
    # stable tie-break on item index for determinism
    order = np.lexsort((part, -masked[part]))
    return part[order]


def ndcg_at_k(ranked_relevance: np.ndarray, n_relevant: int, k: int) -> float:
    """Binary-relevance NDCG@k."""
    if n_relevant <= 0:
        return 0.0
    rel = ranked_relevance[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_len = min(n_relevant, k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_len + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    method: str,
    partition: str,
    score_fn,
    heldout: pd.DataFrame,
    known: pd.DataFrame,
    user_idx: dict[int, int],
    item_idx: dict[int, int],
    candidate_mask: np.ndarray,
    k: int = 10,
    relevance_threshold: float = 4.0,
    batch_size: int = 256,
) -> RankingResult:
    """Compute top-K metrics for ``score_fn`` on ``heldout``.

    Args:
        score_fn: callable(user_row_indices: np.ndarray) -> (len, n_items) scores.
        heldout: ratings of the evaluated partition (validation or test).
        known: ratings the model may know about (to exclude from candidates).
        candidate_mask: boolean (n_items,) serving-policy mask.
    """
    n_items = len(item_idx)
    rel_df = heldout[heldout["rating"] >= relevance_threshold]
    users = sorted(set(rel_df["userId"]) & set(user_idx))
    n_skipped = int(heldout["userId"].nunique() - len(users))

    rel_by_user: dict[int, list[int]] = {}
    not_recommendable = 0
    for uid, grp in rel_df.groupby("userId"):
        cols = [item_idx.get(m, -1) for m in grp["movieId"]]
        not_recommendable += sum(1 for c in cols if c < 0 or not candidate_mask[c])
        rel_by_user[int(uid)] = cols
    known_by_user = {int(uid): np.array([item_idx[m] for m in g["movieId"] if m in item_idx], dtype=np.int64)
                     for uid, g in known.groupby("userId")}

    precisions, recalls, ndcgs, hits = [], [], [], []
    recommended = np.zeros(n_items, dtype=bool)
    for start in range(0, len(users), batch_size):
        chunk = users[start:start + batch_size]
        rows = np.array([user_idx[u] for u in chunk], dtype=np.int64)
        scores = score_fn(rows)
        for r, uid in enumerate(chunk):
            allowed = candidate_mask.copy()
            excl = known_by_user.get(uid)
            if excl is not None and len(excl):
                allowed[excl] = False
            recs = top_n(scores[r], allowed, k)
            recommended[recs] = True
            rel_cols = rel_by_user[uid]
            rel_set = {c for c in rel_cols if c >= 0}
            ranked_rel = np.array([1 if c in rel_set else 0 for c in recs], dtype=int)
            n_rel = len(rel_cols)
            h = int(ranked_rel.sum())
            precisions.append(h / k)
            recalls.append(h / n_rel)
            ndcgs.append(ndcg_at_k(ranked_rel, n_rel, k))
            hits.append(1.0 if h > 0 else 0.0)

    n_candidates = int(candidate_mask.sum())
    res = RankingResult(
        method=method, partition=partition, k=k,
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        catalog_coverage=float(recommended.sum() / n_candidates) if n_candidates else 0.0,
        n_users_evaluated=len(users), n_users_skipped=n_skipped,
        n_relevant_not_recommendable=int(not_recommendable),
    )
    logger.info("%-22s %-10s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f users=%d",
                method, partition, k, res.precision, k, res.recall, k, res.ndcg,
                res.catalog_coverage, res.n_users_evaluated)
    return res


def popularity_scores(item_counts: np.ndarray):
    """Score function ranking every user's candidates by training rating count."""
    base = np.asarray(item_counts, dtype=float)

    def _fn(rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(base, (len(rows), len(base)))

    return _fn


def bias_scores(mu: float, user_bias: np.ndarray, item_bias: np.ndarray):
    """Score function for the additive-bias baseline (mu + b_u + b_i)."""
    def _fn(rows: np.ndarray) -> np.ndarray:
        return mu + user_bias[rows][:, None] + item_bias[None, :]

    return _fn
