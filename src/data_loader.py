"""
data_loader.py
==============
Handles MovieLens dataset download, parsing, and user-item matrix construction.

Mathematical context
--------------------
We construct R in R^(m x n) where R[i, j] = rating of user i on item j.
Unobserved entries are stored as NaN — NOT zero.  Zero is a sentinel value that
would corrupt the Frobenius-norm objective; NaN is explicit about missingness.

Bias decomposition implemented here feeds directly into the optimisation:

    r_{ui} = mu + b_u + b_i + epsilon_{ui}

where epsilon_{ui} is the residual captured by SVD (see svd_model.py).

Equation numbering follows report.tex:
    Eq.(bias_mu)  : mu    = (1/|Omega|) sum_{(u,i) in Omega} r_{ui}
    Eq.(bias_bu)  : b_u   = (1/|Omega_u|) sum_i (r_{ui} - mu)
    Eq.(bias_bi)  : b_i   = (1/|Omega_i|) sum_u (r_{ui} - mu - b_u)
"""

from __future__ import annotations

import logging
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── module-level constants (overridable via config) ───────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR      = Path(__file__).parent.parent / "data"
RATINGS_PATH  = DATA_DIR / "ml-latest-small" / "ratings.csv"
ZIP_PATH      = DATA_DIR / "ml-latest-small.zip"

ML1M_URL          = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_ZIP_PATH     = DATA_DIR / "ml-1m.zip"
ML1M_RATINGS_PATH = DATA_DIR / "ml-1m" / "ratings.dat"


# ── download & extract ────────────────────────────────────────────────────────
def download_movielens(force: bool = False) -> None:
    """Download and unzip the MovieLens latest-small dataset.

    Args:
        force: If True, re-download even if the file already exists.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RATINGS_PATH.exists() and not force:
        logger.info("Dataset already present at %s", RATINGS_PATH)
        return
    logger.info("Downloading MovieLens small dataset from %s", MOVIELENS_URL)
    urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    logger.info("Extraction complete. Ratings file: %s", RATINGS_PATH)


# ── download & extract ML-1M ─────────────────────────────────────────────────
def download_movielens_1m(force: bool = False) -> None:
    """Download and unzip the MovieLens 1M dataset (~24 MB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ML1M_RATINGS_PATH.exists() and not force:
        logger.info("ML-1M already present at %s", ML1M_RATINGS_PATH)
        return
    logger.info("Downloading MovieLens 1M from %s ...", ML1M_URL)
    urllib.request.urlretrieve(ML1M_URL, ML1M_ZIP_PATH)
    with zipfile.ZipFile(ML1M_ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    logger.info("ML-1M extraction complete -> %s", ML1M_RATINGS_PATH)


def load_ratings_ml1m(ratings_path: Optional[Path] = None) -> pd.DataFrame:
    """Load MovieLens 1M ratings (UserID::MovieID::Rating::Timestamp format).

    Ratings are integers in {1, 2, 3, 4, 5}.

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp.
    """
    path = Path(ratings_path) if ratings_path else ML1M_RATINGS_PATH
    if not path.exists():
        download_movielens_1m()
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int64", "movieId": "int64",
               "rating": "float64", "timestamp": "int64"},
    )
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()
    density = len(df) / (n_users * n_items) * 100
    logger.info(
        "Loaded ML-1M: %d ratings | %d users | %d items | density %.2f%%",
        len(df), n_users, n_items, density,
    )
    return df


# ── load raw ratings ──────────────────────────────────────────────────────────
def load_ratings(ratings_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw MovieLens ratings CSV.

    The returned DataFrame has columns:
        userId (int64), movieId (int64), rating (float64), timestamp (int64)

    Ratings are on the scale [0.5, 5.0] in 0.5 increments.

    Args:
        ratings_path: Optional override for the CSV path.  Defaults to the
            module-level RATINGS_PATH.

    Returns:
        Raw ratings DataFrame.

    Raises:
        FileNotFoundError: If the dataset is absent and download fails.
    """
    path = Path(ratings_path) if ratings_path else RATINGS_PATH
    if not path.exists():
        download_movielens()
    df = pd.read_csv(path)

    # -- basic validation --
    required_cols = {"userId", "movieId", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"ratings CSV is missing columns: {missing}")
    if df["rating"].isna().any():
        raise ValueError("ratings CSV contains NaN rating values — data corrupt.")
    if (df["rating"] < 0.5).any() or (df["rating"] > 5.0).any():
        raise ValueError("ratings CSV contains out-of-range values (expected [0.5, 5.0]).")

    logger.info(
        "Loaded %d ratings | %d users | %d items",
        len(df), df["userId"].nunique(), df["movieId"].nunique(),
    )
    return df


# ── train / test split ────────────────────────────────────────────────────────
def train_test_split(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified per-user train/test split.

    For every user u, `test_frac` of their ratings are held out.  This ensures:
    1. Every user appears in both splits (avoids cold-start artefacts in eval).
    2. The test set reflects the realistic distribution of users' activity levels.

    No information from test ratings is used to build the training matrix, so
    there is zero train/test leakage.

    Args:
        df:           Raw ratings DataFrame.
        test_frac:    Fraction [0, 1) of each user's ratings for the test set.
        random_state: NumPy default_rng seed for reproducibility.

    Returns:
        (train_df, test_df): Non-overlapping DataFrames by index.

    Raises:
        ValueError: If test_frac is not in (0, 1).
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    rng = np.random.default_rng(random_state)
    train_rows: list[np.ndarray] = []
    test_rows:  list[np.ndarray] = []

    for _, group in df.groupby("userId"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * test_frac))
        test_rows.append(idx[:cut])
        train_rows.append(idx[cut:])

    train_df = df.loc[np.concatenate(train_rows)].copy().reset_index(drop=True)
    test_df  = df.loc[np.concatenate(test_rows)].copy().reset_index(drop=True)

    # -- leakage check: no (userId, movieId) pair must appear in both splits --
    train_pairs = set(zip(train_df["userId"], train_df["movieId"]))
    test_pairs  = set(zip(test_df["userId"],  test_df["movieId"]))
    overlap_pairs = train_pairs & test_pairs
    assert len(overlap_pairs) == 0, (
        f"BUG: {len(overlap_pairs)} (userId, movieId) pairs appear in both train and test!"
    )

    logger.info(
        "Split -> Train: %d ratings | Test: %d ratings (%.0f%% hold-out)",
        len(train_df), len(test_df), test_frac * 100,
    )
    return train_df, test_df


# ── user-item matrix ──────────────────────────────────────────────────────────
def build_user_item_matrix(
    train_df: pd.DataFrame,
    full_df:  Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Construct the user-item rating matrix R in R^(m x n).

    Mathematical definition (Eq. rating_matrix in report):
        R[i, j] = r_{u_i, v_j}   if (i, j) in Omega  (observed in train_df)
        R[i, j] = NaN             otherwise

    where Omega = {(u, i) : training observation exists}.

    We fix the vocabulary using `full_df` (all ratings, including test) so that
    test-set (userId, movieId) pairs have valid integer indices.  Without this,
    evaluation would silently skip users/items that appeared only in the test set.

    Args:
        train_df: Training ratings DataFrame.
        full_df:  All ratings DataFrame used to define the global vocabulary.
            If None, vocabulary is inferred from train_df only.

    Returns:
        R:         np.ndarray of shape (m, n) with NaN for unobserved entries.
        user_idx:  Mapping userId (int) -> row index i.
        item_idx:  Mapping movieId (int) -> column index j.

    Raises:
        ValueError: If train_df is empty.
    """
    if len(train_df) == 0:
        raise ValueError("train_df is empty — cannot build matrix.")

    ref = full_df if full_df is not None else train_df
    user_ids = sorted(ref["userId"].unique())
    item_ids = sorted(ref["movieId"].unique())
    user_idx: dict[int, int] = {uid: i for i, uid in enumerate(user_ids)}
    item_idx: dict[int, int] = {mid: j for j, mid in enumerate(item_ids)}

    m, n = len(user_ids), len(item_ids)
    R = np.full((m, n), np.nan, dtype=np.float64)

    for row in train_df.itertuples(index=False):
        i = user_idx[row.userId]
        j = item_idx[row.movieId]
        R[i, j] = row.rating

    observed = int(np.count_nonzero(~np.isnan(R)))
    density  = observed / (m * n)
    logger.info(
        "Matrix shape: %d x %d | Observed: %d | Density: %.4f%%",
        m, n, observed, density * 100,
    )
    return R, user_idx, item_idx


# ── mean-centering and bias decomposition ─────────────────────────────────────
def center_matrix(
    R: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Mean-centre R using the additive bias decomposition.

    Implements (Eq. bias_decomp in report):
        r_{ui} = mu + b_u + b_i + epsilon_{ui}

    Step-by-step:
        mu    = (1/|Omega|) sum_{(u,i) in Omega} R[u,i]
        b_u   = nanmean(R - mu, axis=1)                   shape (m,)
        b_i   = nanmean(R - mu - b_u[:, None], axis=0)   shape (n,)
        R_c   = R - mu - b_u[:, None] - b_i[None, :]

    After centering, R_c[u, i] ≈ 0 on average; residual variance is dominated
    by the latent user-item interaction that SVD learns to capture.

    Args:
        R: Raw rating matrix of shape (m, n) with NaN for unobserved entries.

    Returns:
        R_c:       Centered residual matrix, same shape as R.
        mu:        Global mean rating (scalar).
        user_bias: Per-user additive bias, shape (m,).
        item_bias: Per-item additive bias, shape (n,).

    Raises:
        ValueError: If R contains no observed entries.
    """
    if np.all(np.isnan(R)):
        raise ValueError("R has no observed entries — cannot compute mean.")

    mu = float(np.nanmean(R))

    # User bias: mean of (R - mu) per row, over observed entries only
    user_bias: np.ndarray = np.nanmean(R - mu, axis=1)        # shape (m,)
    user_bias = np.where(np.isnan(user_bias), 0.0, user_bias)

    # Item bias: residual mean per column after removing mu and b_u
    R_ub: np.ndarray = R - mu - user_bias[:, np.newaxis]
    item_bias: np.ndarray = np.nanmean(R_ub, axis=0)           # shape (n,)
    item_bias = np.where(np.isnan(item_bias), 0.0, item_bias)

    R_c = R - mu - user_bias[:, np.newaxis] - item_bias[np.newaxis, :]

    logger.debug(
        "Centering: mu=%.4f | max|b_u|=%.4f | max|b_i|=%.4f",
        mu, float(np.max(np.abs(user_bias))), float(np.max(np.abs(item_bias))),
    )
    return R_c, mu, user_bias, item_bias


# ── missing value imputation ──────────────────────────────────────────────────
def impute_matrix(R_c: np.ndarray, strategy: str = "zero") -> np.ndarray:
    """Fill NaN entries so numpy's dense SVD can operate on a complete matrix.

    Connection to optimisation:
        The exact sparse problem  min_{rank(M)<=k} ||P_Omega(R - M)||_F^2
        requires iterative solvers (ALS, SGD-MF).
        For dense SVD we instead minimise  ||R_filled - M||_F^2  on the
        *imputed* complete matrix, which is the Eckart-Young problem and has
        a closed-form solution.  Zero-fill on a mean-centred matrix is the
        standard academically-accepted initialisation for this step.

    Args:
        R_c:      Centred rating matrix (NaN for unobserved entries).
        strategy: Imputation strategy.
            "zero"    - fill NaN with 0 (recommended for mean-centred matrices).
            "rowmean" - fill NaN with the row's mean of observed residuals.
            "colmean" - fill NaN with the column's mean of observed residuals.

    Returns:
        R_filled: Dense matrix with the same shape as R_c, no NaN entries.

    Raises:
        ValueError: If strategy is not one of the supported options.
    """
    valid_strategies = ("zero", "rowmean", "colmean")
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy must be one of {valid_strategies}, got {strategy!r}"
        )

    R_filled = R_c.copy()

    if strategy == "zero":
        R_filled[np.isnan(R_filled)] = 0.0

    elif strategy == "rowmean":
        row_means = np.nanmean(R_c, axis=1, keepdims=True)
        row_means = np.where(np.isnan(row_means), 0.0, row_means)
        mask = np.isnan(R_filled)
        R_filled[mask] = np.broadcast_to(row_means, R_c.shape)[mask]

    else:  # colmean
        col_means = np.nanmean(R_c, axis=0, keepdims=True)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        mask = np.isnan(R_filled)
        R_filled[mask] = np.broadcast_to(col_means, R_c.shape)[mask]

    assert not np.any(np.isnan(R_filled)), "BUG: NaN values remain after imputation."
    logger.debug("Imputation (%s): matrix fully dense, shape %s", strategy, R_filled.shape)
    return R_filled
