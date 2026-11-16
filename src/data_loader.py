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

import hashlib
import logging
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
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

ML_SMALL_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
EXTRA_CA_FILE = Path(__file__).parent.parent / "certs" / "incommon-ovg2c-intermediate.pem"

ML1M_URL          = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_ZIP_PATH     = DATA_DIR / "ml-1m.zip"
ML1M_RATINGS_PATH = DATA_DIR / "ml-1m" / "ratings.dat"


# ── download & extract ────────────────────────────────────────────────────────
def _ssl_context(extra_ca_file: Optional[Path] = None) -> ssl.SSLContext:
    """Default (verifying) TLS context, optionally extended with an extra CA file.

    files.grouplens.org serves its leaf certificate without the intermediate.
    Browsers fetch the intermediate automatically; Python does not, so the public
    intermediate is shipped in ``certs/`` and added here. Verification is never
    disabled.
    """
    ctx = ssl.create_default_context()
    if extra_ca_file is not None and Path(extra_ca_file).exists():
        ctx.load_verify_locations(cafile=str(extra_ca_file))
        # The intermediate's own root (emSign Root TLS CA - G1) is missing from some
        # OS trust stores. PARTIAL_CHAIN lets the pinned intermediate act as a trust
        # anchor (Python 3.13 enables this flag by default). Hostname and signature
        # checks are unchanged; CERT_REQUIRED stays in force.
        ctx.verify_flags |= getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    return ctx


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(
    url: str,
    dest_zip: Path,
    expected_sha256: Optional[str] = None,
    extra_ca_file: Optional[Path] = None,
    timeout: float = 120.0,
) -> Path:
    """Download ``url`` to ``dest_zip`` atomically and verify its checksum.

    Raises:
        urllib.error.URLError: network or TLS failure.
        ValueError: the downloaded file does not match ``expected_sha256``.
    """
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context(extra_ca_file)
    logger.info("Downloading %s", url)
    with tempfile.NamedTemporaryFile(dir=dest_zip.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=timeout) as resp:  # noqa: S310
                shutil.copyfileobj(resp, tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    digest = sha256_file(tmp_path)
    if expected_sha256 and digest != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, got {digest}"
        )
    tmp_path.replace(dest_zip)
    logger.info("Downloaded %s (sha256 %s)", dest_zip, digest[:16])
    return dest_zip


def ensure_dataset(cfg: dict) -> Path:
    """Make sure the configured ratings file exists, downloading it if needed.

    Returns:
        Path to the ratings CSV.
    """
    from settings import resolve_path

    ds = cfg.get("dataset", {})
    ratings_path = resolve_path(ds.get("ratings_path", "data/ml-latest-small/ratings.csv"))
    if ratings_path.exists():
        return ratings_path
    data_dir = resolve_path(ds.get("local_dir", "data"))
    zip_path = data_dir / ds.get("zip_filename", "ml-latest-small.zip")
    extra_ca = ds.get("extra_ca_file")
    if not zip_path.exists():
        download_archive(
            ds.get("url", MOVIELENS_URL), zip_path,
            expected_sha256=ds.get("zip_sha256"),
            extra_ca_file=resolve_path(extra_ca) if extra_ca else None,
        )
    elif ds.get("zip_sha256") and sha256_file(zip_path) != ds["zip_sha256"]:
        raise ValueError(f"{zip_path} exists but does not match the configured checksum.")
    with zipfile.ZipFile(zip_path, "r") as z:
        _safe_extract(z, data_dir)
    if not ratings_path.exists():
        raise FileNotFoundError(f"Archive extracted but {ratings_path} is missing.")
    return ratings_path


def _safe_extract(z: zipfile.ZipFile, dest: Path) -> None:
    """Extract an archive, refusing members that would escape ``dest``."""
    dest = dest.resolve()
    for member in z.namelist():
        target = (dest / member).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(f"Unsafe path in archive: {member}")
    z.extractall(dest)


def download_movielens(force: bool = False) -> None:
    """Download and unzip the MovieLens latest-small dataset (legacy helper).

    Uses the same verified, TLS-checked path as :func:`ensure_dataset`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RATINGS_PATH.exists() and not force:
        logger.info("Dataset already present at %s", RATINGS_PATH)
        return
    download_archive(MOVIELENS_URL, ZIP_PATH, expected_sha256=ML_SMALL_SHA256,
                     extra_ca_file=EXTRA_CA_FILE)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        _safe_extract(z, DATA_DIR)
    logger.info("Extraction complete. Ratings file: %s", RATINGS_PATH)


# ── download & extract ML-1M ─────────────────────────────────────────────────
def download_movielens_1m(force: bool = False) -> None:
    """Download and unzip the MovieLens 1M dataset (~24 MB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ML1M_RATINGS_PATH.exists() and not force:
        logger.info("ML-1M already present at %s", ML1M_RATINGS_PATH)
        return
    download_archive(ML1M_URL, ML1M_ZIP_PATH, extra_ca_file=EXTRA_CA_FILE)
    with zipfile.ZipFile(ML1M_ZIP_PATH, "r") as z:
        _safe_extract(z, DATA_DIR)
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
        # .copy(): under pandas >= 3 (copy-on-write) index arrays are read-only and
        # an in-place shuffle raises "ValueError: array is read-only".
        idx = group.index.to_numpy().copy()
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
    if overlap_pairs:
        raise RuntimeError(
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

    if np.any(np.isnan(R_filled)):
        raise RuntimeError("BUG: NaN values remain after imputation.")
    logger.debug("Imputation (%s): matrix fully dense, shape %s", strategy, R_filled.shape)
    return R_filled


# ═══════════════════════════════════════════════════════════════════════════════
# Production data layer: validation, fingerprinting, leakage-safe 3-way split
# ═══════════════════════════════════════════════════════════════════════════════
RATING_COLUMNS = ("userId", "movieId", "rating")


@dataclass
class DataQualityReport:
    """Outcome of :func:`validate_ratings`. ``passed`` is False on any hard error."""

    n_input_rows: int = 0
    n_output_rows: int = 0
    n_users: int = 0
    n_items: int = 0
    n_missing_values: int = 0
    n_out_of_range: int = 0
    n_non_integer_ids: int = 0
    n_off_scale_step: int = 0
    n_duplicate_pairs: int = 0
    duplicates_policy: str = "keep_latest"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


class DataValidationError(ValueError):
    """Raised when ratings data fails a hard validation rule."""

    def __init__(self, report: DataQualityReport):
        super().__init__("; ".join(report.errors))
        self.report = report


def validate_ratings(
    df: pd.DataFrame,
    rating_scale: tuple[float, float] = (0.5, 5.0),
    rating_step: Optional[float] = 0.5,
    drop_invalid: bool = False,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and normalise a ratings table.

    Hard errors (raise unless ``drop_invalid``): missing required columns, missing
    values, non-integer or non-positive ids, ratings outside ``rating_scale``.
    Duplicate (userId, movieId) pairs are resolved by keeping the latest rating
    (by ``timestamp`` when present) and are always reported.

    Args:
        df: Raw ratings with at least userId, movieId, rating.
        rating_scale: Inclusive (min, max).
        rating_step: Expected granularity; off-step values are reported as a warning.
        drop_invalid: Drop invalid rows instead of raising (used for feedback data).

    Returns:
        (clean_df, report). ``clean_df`` has int64 ids and float64 ratings.

    Raises:
        DataValidationError: On hard errors when ``drop_invalid`` is False.
    """
    rep = DataQualityReport(n_input_rows=int(len(df)))
    missing_cols = [c for c in RATING_COLUMNS if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing required columns: {missing_cols}")
        raise DataValidationError(rep)

    out = df.copy()
    cols = list(RATING_COLUMNS) + (["timestamp"] if "timestamp" in out.columns else [])
    out = out[cols]

    na_mask = out[list(RATING_COLUMNS)].isna().any(axis=1)
    rep.n_missing_values = int(na_mask.sum())

    ids = out[["userId", "movieId"]].apply(pd.to_numeric, errors="coerce")
    bad_id = ids.isna().any(axis=1) | (ids % 1 != 0).any(axis=1) | (ids <= 0).any(axis=1)
    rep.n_non_integer_ids = int((bad_id & ~na_mask).sum())

    ratings = pd.to_numeric(out["rating"], errors="coerce")
    lo, hi = float(rating_scale[0]), float(rating_scale[1])
    out_of_range = ratings.isna() | (ratings < lo) | (ratings > hi)
    rep.n_out_of_range = int((out_of_range & ~na_mask).sum())

    invalid = na_mask | bad_id | out_of_range
    if invalid.any():
        msg = (f"{int(invalid.sum())} invalid rows (missing={rep.n_missing_values}, "
               f"bad_ids={rep.n_non_integer_ids}, out_of_range={rep.n_out_of_range})")
        if drop_invalid:
            rep.warnings.append(msg + " dropped")
        else:
            rep.errors.append(msg)
            raise DataValidationError(rep)
    out = out.loc[~invalid].copy()
    out["userId"] = ids.loc[out.index, "userId"].astype("int64")
    out["movieId"] = ids.loc[out.index, "movieId"].astype("int64")
    out["rating"] = ratings.loc[out.index].astype("float64")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype("int64")

    if rating_step:
        off = ((out["rating"] / rating_step) % 1 != 0)
        rep.n_off_scale_step = int(off.sum())
        if rep.n_off_scale_step:
            rep.warnings.append(f"{rep.n_off_scale_step} ratings are not multiples of {rating_step}")

    dup = out.duplicated(["userId", "movieId"], keep=False)
    rep.n_duplicate_pairs = int(out.duplicated(["userId", "movieId"]).sum())
    if dup.any():
        sort_cols = ["userId", "movieId"] + (["timestamp"] if "timestamp" in out.columns else [])
        out = out.sort_values(sort_cols, kind="mergesort").drop_duplicates(
            ["userId", "movieId"], keep="last")
        rep.warnings.append(f"{rep.n_duplicate_pairs} duplicate (userId, movieId) rows resolved "
                            f"by keeping the latest rating")

    out = out.reset_index(drop=True)
    rep.n_output_rows = int(len(out))
    rep.n_users = int(out["userId"].nunique())
    rep.n_items = int(out["movieId"].nunique())
    if rep.n_output_rows == 0:
        rep.errors.append("no valid rows")
        if not drop_invalid:
            raise DataValidationError(rep)
    return out, rep


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent SHA-256 of the (userId, movieId, rating) content."""
    canon = df[list(RATING_COLUMNS)].sort_values(["userId", "movieId"], kind="mergesort")
    payload = canon.to_csv(index=False, lineterminator="\n", float_format="%.4f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_ratings_from_config(cfg: dict, path: Optional[Path] = None) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load (downloading if necessary) and validate the configured ratings file."""
    ratings_path = Path(path) if path else ensure_dataset(cfg)
    raw = pd.read_csv(ratings_path)
    scale = tuple(cfg.get("dataset", {}).get("rating_scale", [0.5, 5.0]))
    df, rep = validate_ratings(raw, rating_scale=scale)
    logger.info("Loaded %d ratings | %d users | %d items | duplicates resolved: %d",
                rep.n_output_rows, rep.n_users, rep.n_items, rep.n_duplicate_pairs)
    return df, rep


def load_movies(path: Path) -> pd.DataFrame:
    """Load MovieLens ``movies.csv`` (movieId, title, genres). Missing file -> empty frame."""
    path = Path(path)
    if not path.exists():
        logger.warning("Movie metadata not found at %s; responses will omit titles.", path)
        return pd.DataFrame(columns=["movieId", "title", "genres"])
    movies = pd.read_csv(path)
    missing = {"movieId", "title", "genres"} - set(movies.columns)
    if missing:
        raise ValueError(f"movies file {path} is missing columns {missing}")
    movies = movies.drop_duplicates("movieId", keep="first")
    movies["movieId"] = movies["movieId"].astype("int64")
    return movies[["movieId", "title", "genres"]].reset_index(drop=True)


@dataclass
class SplitResult:
    """Train / validation / test partitions plus their integrity record."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict


def train_val_test_split(
    df: pd.DataFrame,
    validation_frac: float = 0.10,
    test_frac: float = 0.20,
    random_state: int = 42,
) -> SplitResult:
    """Leakage-safe, per-user stratified three-way split.

    For each user, ratings are shuffled with a seeded RNG (users are processed in
    sorted order, so the result is independent of input row order). Then
    ``round(test_frac * n)`` (at least 1) go to test, ``round(validation_frac * n)``
    (at least 1) to validation and the remainder (at least 1) to train. Users with
    fewer than 3 ratings are kept entirely in train and reported.

    Guarantees (checked, raises on violation):
      * every (userId, movieId) pair lives in exactly one partition;
      * partitions are disjoint and their union is the input.

    The validation partition is for model selection; the test partition is for a
    single final evaluation after selection.
    """
    if not 0.0 < test_frac < 1.0 or not 0.0 < validation_frac < 1.0:
        raise ValueError("fractions must be in (0, 1)")
    if test_frac + validation_frac >= 0.9:
        raise ValueError("test_frac + validation_frac must be < 0.9")
    if df.duplicated(["userId", "movieId"]).any():
        raise ValueError("duplicate (userId, movieId) pairs: run validate_ratings() first")

    rng = np.random.default_rng(random_state)
    canon = df.sort_values(["userId", "movieId"], kind="mergesort").reset_index(drop=True)
    part = np.empty(len(canon), dtype=np.int8)  # 0 train, 1 validation, 2 test
    n_small_users = 0
    starts = np.flatnonzero(np.r_[True, canon["userId"].to_numpy()[1:] != canon["userId"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(canon)]
    for s0, e0 in zip(starts, ends):
        n = int(e0 - s0)
        order = rng.permutation(n)
        labels = np.zeros(n, dtype=np.int8)
        if n >= 3:
            n_test = max(1, int(round(test_frac * n)))
            n_val = max(1, int(round(validation_frac * n)))
            if n_test + n_val > n - 1:
                n_val = max(1, n - 1 - n_test)
            labels[order[:n_test]] = 2
            labels[order[n_test:n_test + n_val]] = 1
        else:
            n_small_users += 1
        part[s0:e0] = labels

    train = canon.loc[part == 0].reset_index(drop=True)
    val = canon.loc[part == 1].reset_index(drop=True)
    test = canon.loc[part == 2].reset_index(drop=True)
    leak = verify_no_leakage(train, val, test)
    if not leak["passed"]:
        raise RuntimeError(f"Split leakage detected: {leak}")
    assignment = canon[["userId", "movieId"]].assign(partition=part)
    manifest = {
        "method": "per-user stratified random split (seeded)",
        "random_state": int(random_state),
        "validation_fraction": float(validation_frac),
        "test_fraction": float(test_frac),
        "n_rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "n_users": {"train": int(train["userId"].nunique()),
                    "validation": int(val["userId"].nunique()),
                    "test": int(test["userId"].nunique())},
        "n_items": {"train": int(train["movieId"].nunique()),
                    "validation": int(val["movieId"].nunique()),
                    "test": int(test["movieId"].nunique())},
        "users_kept_in_train_only": int(n_small_users),
        "items_in_validation_not_in_train": int(len(set(val["movieId"]) - set(train["movieId"]))),
        "items_in_test_not_in_train": int(len(set(test["movieId"]) - set(train["movieId"]))),
        "split_sha256": hashlib.sha256(
            assignment.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest(),
        "dataset_sha256": dataset_fingerprint(df),
        "leakage_checks": leak,
    }
    logger.info("Split -> train %d | validation %d | test %d (users kept train-only: %d)",
                len(train), len(val), len(test), n_small_users)
    return SplitResult(train=train, validation=val, test=test, manifest=manifest)


def verify_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Count (userId, movieId) pairs shared between partitions (must all be 0)."""
    def pairs(d: pd.DataFrame) -> set[tuple[int, int]]:
        return set(zip(d["userId"].tolist(), d["movieId"].tolist()))

    tr, va, te = pairs(train), pairs(val), pairs(test)
    res = {
        "train_validation_shared_pairs": len(tr & va),
        "train_test_shared_pairs": len(tr & te),
        "validation_test_shared_pairs": len(va & te),
        "duplicate_pairs_within_partitions": int(
            train.duplicated(["userId", "movieId"]).sum()
            + val.duplicated(["userId", "movieId"]).sum()
            + test.duplicated(["userId", "movieId"]).sum()),
    }
    res["passed"] = all(v == 0 for v in res.values())
    return res
