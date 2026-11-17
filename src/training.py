"""
training.py
===========
Reusable training and model-selection core, shared by ``main.py`` (initial
pipeline) and ``retrain.py`` (controlled retraining).

Selection protocol (the correction of the historical test-set leakage):

1. Fit on the TRAIN partition only (vocabulary = users/items seen in train).
2. Sweep the SVD rank ``k`` and score every candidate on VALIDATION.
3. Pick ``k`` with the lowest validation RMSE (ties -> smaller k).
4. Only after that, evaluate the chosen configuration once on TEST.

``select_rank`` never receives the test partition, so it cannot use it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import build_user_item_matrix, center_matrix, impute_matrix
from evaluation import evaluate_pairs
from ranking import evaluate_ranking
from svd_model import SVDModel, eckart_young_error

logger = logging.getLogger(__name__)

RANDOMIZED_THRESHOLD = 3000   # use randomized SVD when min(m, n) exceeds this


@dataclass
class Decomposition:
    """Bias terms plus the (possibly truncated) SVD spectrum of the train matrix."""

    R: np.ndarray
    user_idx: dict[int, int]
    item_idx: dict[int, int]
    user_ids: np.ndarray
    item_ids: np.ndarray
    mu: float
    user_bias: np.ndarray
    item_bias: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    solver: str
    seconds: float

    def model(self, k: int) -> SVDModel:
        k_eff = int(min(k, len(self.S)))
        return SVDModel(U_k=self.U[:, :k_eff], sigma_k=self.S[:k_eff], Vt_k=self.Vt[:k_eff, :],
                        mu=self.mu, user_bias=self.user_bias, item_bias=self.item_bias,
                        all_singular_values=self.S, k=k_eff)

    @property
    def item_counts(self) -> np.ndarray:
        return np.count_nonzero(~np.isnan(self.R), axis=0)


def decompose(train_df: pd.DataFrame, k_max: int, imputation: str = "zero",
              solver: str = "auto", random_state: int = 42) -> Decomposition:
    """Build the train matrix, bias-centre it, impute and compute its SVD once."""
    R, user_idx, item_idx = build_user_item_matrix(train_df)   # train vocabulary only
    R_c, mu, ub, ib = center_matrix(R)
    R_filled = impute_matrix(R_c, strategy=imputation)
    m, n = R_filled.shape
    use_rand = solver == "randomized" or (solver == "auto" and min(m, n) > RANDOMIZED_THRESHOLD)
    t0 = time.perf_counter()
    if use_rand:
        from sklearn.utils.extmath import randomized_svd
        n_comp = int(min(min(m, n) - 1, k_max + 20))
        U, S, Vt = randomized_svd(R_filled, n_components=n_comp, n_iter=4,
                                  random_state=random_state)
        used = f"randomized(n_components={n_comp})"
    else:
        U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)
        used = "full"
    secs = time.perf_counter() - t0
    logger.info("SVD of %dx%d train matrix via %s in %.2fs", m, n, used, secs)
    inv_u = np.array(sorted(user_idx, key=user_idx.get), dtype=np.int64)
    inv_i = np.array(sorted(item_idx, key=item_idx.get), dtype=np.int64)
    return Decomposition(R=R, user_idx=user_idx, item_idx=item_idx, user_ids=inv_u, item_ids=inv_i,
                         mu=mu, user_bias=ub, item_bias=ib, U=U, S=S, Vt=Vt,
                         solver=used, seconds=secs)


def select_rank(dec: Decomposition, k_values: list[int], validation_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Sweep ``k`` on the VALIDATION partition. Returns (best_k, sweep table)."""
    rows = []
    m, n = dec.R.shape
    for k in k_values:
        model = dec.model(k)
        res = evaluate_pairs(f"SVD k={k}", model, validation_df, dec.user_idx, dec.item_idx)
        ey = eckart_young_error(dec.S, model.k) / np.sqrt(m * n) if dec.solver == "full" else np.nan
        ev = float(np.sum(dec.S[:model.k] ** 2) / np.sum(dec.S ** 2))
        rows.append({"k": int(k), "k_effective": int(model.k), "val_rmse": res.rmse,
                     "val_mae": res.mae, "n": res.n_samples, "n_fallback": res.n_fallback,
                     "explained_variance": ev, "ey_bound_normalised": ey})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["val_rmse", "k"], kind="mergesort").iloc[0]
    logger.info("Validation-selected rank k=%d (validation RMSE %.4f)", int(best["k"]), best["val_rmse"])
    return int(best["k"]), sweep


def candidate_mask(item_counts: np.ndarray, min_item_ratings: int) -> np.ndarray:
    return np.asarray(item_counts) >= int(min_item_ratings)


def rating_and_ranking_metrics(name: str, model, partition: str, heldout: pd.DataFrame,
                               known: pd.DataFrame, dec: Decomposition, score_fn,
                               rank_cfg: dict) -> dict:
    """RMSE/MAE plus top-K metrics for one model on one partition."""
    pr = evaluate_pairs(name, model, heldout, dec.user_idx, dec.item_idx)
    out = {"model": name, "partition": partition, "rmse": pr.rmse, "mae": pr.mae,
           "n": pr.n_samples, "n_fallback": pr.n_fallback}
    if score_fn is not None:
        rr = evaluate_ranking(name, partition, score_fn, heldout, known, dec.user_idx,
                              dec.item_idx,
                              candidate_mask(dec.item_counts, rank_cfg.get("min_item_ratings", 5)),
                              k=int(rank_cfg.get("k", 10)),
                              relevance_threshold=float(rank_cfg.get("relevance_threshold", 4.0)))
        out.update({f"precision@{rr.k}": rr.precision, f"recall@{rr.k}": rr.recall,
                    f"ndcg@{rr.k}": rr.ndcg, f"hit_rate@{rr.k}": rr.hit_rate,
                    "catalog_coverage": rr.catalog_coverage,
                    "ranking_users": rr.n_users_evaluated})
    return out
