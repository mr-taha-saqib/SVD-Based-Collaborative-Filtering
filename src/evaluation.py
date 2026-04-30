"""
evaluation.py
=============
Evaluation metrics and baseline models.

Metric definitions (Eq. metrics in report)
-------------------------------------------
RMSE:   Root Mean Squared Error
    RMSE = sqrt( (1/|T|) * sum_{(u,i) in T} (r_{ui} - r_hat_{ui})^2 )

    Penalises large errors quadratically.  Appropriate when a 1-star miss is
    much worse than five 0.2-star misses.

MAE:    Mean Absolute Error
    MAE = (1/|T|) * sum_{(u,i) in T} |r_{ui} - r_hat_{ui}|

    Linear in error magnitude; more robust to outliers than RMSE.
    Easier to interpret in rating-scale units (e.g. "off by 0.67 stars").

Baseline hierarchy
------------------
1. GlobalMean    : r_hat = mu                   (intercept only)
2. UserMean      : r_hat = mu_u                 (user intercept)
3. ItemMean      : r_hat = mu_i                 (item intercept)
4. UserItemBias  : r_hat = mu + b_u + b_i       (exact SVD de-centering; fairest baseline)
5. SGDMatrixFact : r_hat = mu + b_u + b_i + p_u . q_i  (sparse-aware, with regularisation)

Comparison rationale:
    Each baseline removes one layer of structure from the target signal.
    The difference SVD - UserItemBias in RMSE quantifies ONLY the contribution
    of the latent factors U_k Sigma_k Vt_k beyond additive biases.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── metric functions ──────────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt( (1/N) * sum_{t=1}^{N} (y_t - y_hat_t)^2 )

    Args:
        y_true: Ground-truth ratings, shape (N,).
        y_pred: Predicted ratings,    shape (N,).

    Returns:
        RMSE as a scalar float.

    Raises:
        ValueError: If arrays have different shapes or are empty.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute RMSE on empty arrays.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    MAE = (1/N) * sum_{t=1}^{N} |y_t - y_hat_t|

    Args:
        y_true: Ground-truth ratings, shape (N,).
        y_pred: Predicted ratings,    shape (N,).

    Returns:
        MAE as a scalar float.

    Raises:
        ValueError: If arrays have different shapes or are empty.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute MAE on empty arrays.")
    return float(np.mean(np.abs(y_true - y_pred)))


def coverage(y_pred: np.ndarray) -> float:
    """Fraction of predictions that are finite (not NaN or inf).

    Args:
        y_pred: Array of predictions.

    Returns:
        Coverage fraction in [0, 1].
    """
    return float(np.mean(np.isfinite(np.asarray(y_pred, float))))


# ── evaluation result container ───────────────────────────────────────────────
@dataclass
class EvalResult:
    """Stores evaluation metrics for one model.

    Attributes:
        model_name: Human-readable model identifier.
        rmse:       Root Mean Squared Error on the test set.
        mae:        Mean Absolute Error on the test set.
        n_samples:  Number of test pairs evaluated.
        elapsed_s:  Wall-clock time for prediction (seconds).
    """
    model_name: str
    rmse:       float
    mae:        float
    n_samples:  int
    elapsed_s:  float = 0.0


# ── evaluation harness ────────────────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    predict_fn: Callable[[int, int], float],
    test_df:    pd.DataFrame,
    user_idx:   dict[int, int],
    item_idx:   dict[int, int],
) -> EvalResult:
    """Evaluate a prediction function on the held-out test split.

    Iterates over all (userId, movieId, rating) triples in test_df, converts
    IDs to matrix indices, calls predict_fn, and accumulates ground-truth and
    predicted arrays for metric computation.

    Cold-start handling: if a userId or movieId is absent from the training
    vocabulary (user_idx / item_idx), that test entry is silently skipped.
    The fraction skipped should be ~0 for a well-formed stratified split.

    Args:
        model_name: Label used in log output and result tables.
        predict_fn: Callable (user_row_idx: int, item_col_idx: int) -> float.
        test_df:    DataFrame with columns [userId, movieId, rating].
        user_idx:   userId -> row index mapping from build_user_item_matrix.
        item_idx:   movieId -> col index mapping from build_user_item_matrix.

    Returns:
        EvalResult with RMSE, MAE, N, and elapsed time.
    """
    y_true: list[float] = []
    y_pred: list[float] = []
    skipped = 0

    t0 = time.perf_counter()
    for row in test_df.itertuples(index=False):
        u = user_idx.get(row.userId)
        v = item_idx.get(row.movieId)
        if u is None or v is None:
            skipped += 1
            continue
        y_true.append(float(row.rating))
        y_pred.append(float(predict_fn(u, v)))

    elapsed = time.perf_counter() - t0

    arr_true = np.array(y_true, dtype=np.float64)
    arr_pred = np.array(y_pred, dtype=np.float64)

    r = rmse(arr_true, arr_pred)
    m = mae(arr_true, arr_pred)

    if skipped:
        logger.warning(
            "%s: skipped %d/%d test entries (cold-start users/items)",
            model_name, skipped, len(test_df),
        )
    logger.info(
        "%-30s  RMSE=%.4f  MAE=%.4f  N=%d  (%.2fs)",
        model_name, r, m, len(arr_true), elapsed,
    )
    return EvalResult(
        model_name=model_name, rmse=r, mae=m,
        n_samples=len(arr_true), elapsed_s=elapsed,
    )


# ── baselines ─────────────────────────────────────────────────────────────────
class GlobalMeanBaseline:
    """Global mean baseline: r_hat_{ui} = mu.

    Optimal under squared loss when all ratings are i.i.d. from the same
    distribution.  Provides a lower-bound sanity check — any model that
    cannot beat this has learned nothing from the data structure.
    """

    def __init__(self) -> None:
        self.mu: Optional[float] = None

    def fit(self, R: np.ndarray) -> "GlobalMeanBaseline":
        """Fit by computing the global mean of observed entries.

        Args:
            R: Rating matrix with NaN for unobserved entries.

        Returns:
            Self (fluent interface).
        """
        self.mu = float(np.nanmean(R))
        logger.debug("GlobalMean: mu=%.4f", self.mu)
        return self

    def predict(self, u: int, i: int) -> float:
        """Return the global mean regardless of (u, i)."""
        return self.mu  # type: ignore[return-value]

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised version of predict."""
        return np.full(len(users), self.mu)


class UserMeanBaseline:
    """User mean baseline: r_hat_{ui} = mu_u.

    Falls back to global mean for users with no training ratings.
    Captures user-level leniency/severity bias but ignores item effects.
    """

    def __init__(self) -> None:
        self.user_means: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None

    def fit(self, R: np.ndarray) -> "UserMeanBaseline":
        """Fit per-user means from observed entries.

        Args:
            R: Rating matrix (m x n) with NaN for unobserved.

        Returns:
            Self.
        """
        self.global_mean = float(np.nanmean(R))
        row_means = np.nanmean(R, axis=1)
        self.user_means = np.where(np.isnan(row_means), self.global_mean, row_means)
        logger.debug("UserMean: mean of user_means=%.4f", float(np.mean(self.user_means)))
        return self

    def predict(self, u: int, i: int) -> float:
        """Return the mean rating for user u."""
        return float(self.user_means[u])  # type: ignore[index]

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised version of predict."""
        return self.user_means[users]  # type: ignore[index]


class ItemMeanBaseline:
    """Item mean baseline: r_hat_{ui} = mu_i.

    Captures item-level popularity bias.  Items rated higher on average
    (blockbusters, award-winners) have higher predicted scores for all users.
    Falls back to global mean for unrated items.
    """

    def __init__(self) -> None:
        self.item_means: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None

    def fit(self, R: np.ndarray) -> "ItemMeanBaseline":
        """Fit per-item means from observed entries.

        Args:
            R: Rating matrix (m x n) with NaN for unobserved.

        Returns:
            Self.
        """
        self.global_mean = float(np.nanmean(R))
        col_means = np.nanmean(R, axis=0)
        self.item_means = np.where(np.isnan(col_means), self.global_mean, col_means)
        logger.debug("ItemMean: mean of item_means=%.4f", float(np.mean(self.item_means)))
        return self

    def predict(self, u: int, i: int) -> float:
        """Return the mean rating for item i."""
        return float(self.item_means[i])  # type: ignore[index]

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised version of predict."""
        return self.item_means[items]  # type: ignore[index]


class UserItemMeanBaseline:
    """Additive bias model: r_hat_{ui} = mu + b_u + b_i.

    This is the fairest baseline for SVD because:
    1. It uses exactly the same bias terms (mu, b_u, b_i) that SVD de-centres with.
    2. The RMSE gap  SVD - UserItemBias  measures PURELY the contribution of the
       latent factors U_k Sigma_k Vt_k, with no confounding from bias differences.
    """

    def __init__(self) -> None:
        self.mu: Optional[float] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None

    def fit(self, R: np.ndarray) -> "UserItemMeanBaseline":
        """Fit bias terms using the same center_matrix procedure as SVD.

        Args:
            R: Rating matrix (m x n) with NaN for unobserved.

        Returns:
            Self.
        """
        from data_loader import center_matrix
        _, self.mu, self.user_bias, self.item_bias = center_matrix(R)
        logger.debug("UserItemBias: mu=%.4f", self.mu)
        return self

    def predict(self, u: int, i: int) -> float:
        """Return mu + b_u + b_i, clipped to [0.5, 5.0]."""
        return float(np.clip(self.mu + self.user_bias[u] + self.item_bias[i], 0.5, 5.0))  # type: ignore

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised version of predict."""
        preds = self.mu + self.user_bias[users] + self.item_bias[items]  # type: ignore
        return np.clip(preds, 0.5, 5.0)


# ── SGD matrix factorisation ──────────────────────────────────────────────────
class SGDMatrixFactorization:
    """SGD-based matrix factorisation (Simon Funk / Netflix Prize style).

    Optimisation problem (Eq. sgd in report):
        min_{P, Q, b_u, b_i}
            sum_{(u,i) in Omega} (r_{ui} - mu - b_u - b_i - P[u].Q[i])^2
            + lambda * (||P||_F^2 + ||Q||_F^2 + ||b_u||^2 + ||b_i||^2)

    SGD update rules (derived from gradient of the objective):
        e   = r_{ui} - r_hat_{ui}                       (residual)
        P[u] <- P[u] + alpha * (e * Q[i] - lambda * P[u])
        Q[i] <- Q[i] + alpha * (e * P[u] - lambda * Q[i])
        b_u  <- b_u  + alpha * (e          - lambda * b_u)
        b_i  <- b_i  + alpha * (e          - lambda * b_i)

    Key difference from SVD:
        SVD minimises ||R_filled - M||_F^2 on the fully imputed matrix
        (solved in closed form via Eckart-Young).
        SGD-MF minimises the same quadratic loss ONLY on Omega (observed entries),
        avoiding imputation bias.  This is why SGD-MF achieves slightly lower RMSE
        despite having more hyperparameters and no closed-form solution.

    Attributes:
        k:        Latent dimension.
        lr:       Learning rate (alpha).
        reg:      L2 regularisation coefficient (lambda).
        epochs:   Number of passes over the training data.
        clip:     (min, max) bounds for predicted ratings.
    """

    def __init__(
        self,
        k:            int   = 20,
        lr:           float = 0.005,
        reg:          float = 0.02,
        epochs:       int   = 20,
        clip:         tuple[float, float] = (0.5, 5.0),
        log_interval: int   = 5,
    ) -> None:
        self.k            = k
        self.lr           = lr
        self.reg          = reg
        self.epochs       = epochs
        self.clip         = clip
        self.log_interval = log_interval

    def fit(self, R: np.ndarray) -> "SGDMatrixFactorization":
        """Train SGD-MF on the observed entries of R.

        Args:
            R: Rating matrix (m x n) with NaN for unobserved entries.

        Returns:
            Self (fitted model).
        """
        m, n = R.shape
        rng  = np.random.default_rng(42)

        self.mu = float(np.nanmean(R))
        self.bu = np.zeros(m, dtype=np.float64)
        self.bi = np.zeros(n, dtype=np.float64)
        self.P  = rng.normal(0.0, 0.01, (m, self.k))
        self.Q  = rng.normal(0.0, 0.01, (n, self.k))

        obs_rows, obs_cols = np.where(~np.isnan(R))
        obs_vals = R[obs_rows, obs_cols]
        order    = np.arange(len(obs_rows))

        logger.info(
            "SGD-MF training: k=%d  lr=%.4f  reg=%.4f  epochs=%d  |Omega|=%d",
            self.k, self.lr, self.reg, self.epochs, len(obs_rows),
        )
        for ep in range(self.epochs):
            rng.shuffle(order)
            total_sq = 0.0
            for idx in order:
                u, i = int(obs_rows[idx]), int(obs_cols[idx])
                r    = obs_vals[idx]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                e    = r - pred
                total_sq += e * e

                Pu_old = self.P[u].copy()
                self.P[u]  += self.lr * (e * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (e * Pu_old    - self.reg * self.Q[i])
                self.bu[u] += self.lr * (e              - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e              - self.reg * self.bi[i])

            if (ep + 1) % self.log_interval == 0:
                train_rmse = float(np.sqrt(total_sq / len(order)))
                logger.info("SGD-MF epoch %3d/%d  train RMSE=%.4f", ep + 1, self.epochs, train_rmse)

        return self

    def predict(self, u: int, i: int) -> float:
        """Predict the rating for user u, item i."""
        p = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(p, *self.clip))

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised prediction for arrays of user/item indices."""
        preds = (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.einsum("nk,nk->n", self.P[users], self.Q[items])
        )
        return np.clip(preds, *self.clip)


# ── results table ──────────────────────────────────────────────────────────────
def results_table(results: list[EvalResult]) -> pd.DataFrame:
    """Build a tidy summary DataFrame sorted by RMSE (ascending).

    Args:
        results: List of EvalResult objects from evaluate_model().

    Returns:
        DataFrame with columns [Model, RMSE, MAE, N, Time_s].
    """
    rows = [
        {
            "Model":  r.model_name,
            "RMSE":   round(r.rmse,      4),
            "MAE":    round(r.mae,       4),
            "N":      r.n_samples,
            "Time_s": round(r.elapsed_s, 3),
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("RMSE").reset_index(drop=True)
