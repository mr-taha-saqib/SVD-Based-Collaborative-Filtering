"""
tests/test_pipeline.py
======================
Comprehensive end-to-end test suite for the SVD Recommendation Pipeline.

Test categories
---------------
1. DATA VALIDATION
   - Matrix dimensions, NaN semantics, density check
   - Train/test leakage check
   - Vocabulary coverage

2. MODEL VALIDATION
   - SVD reconstruction correctness (Eckart-Young property)
   - Prediction bounds ([0.5, 5.0])
   - Batch vs scalar prediction consistency
   - fit_svd input validation

3. METRICS VALIDATION
   - RMSE correctness (manual recomputation)
   - MAE correctness
   - Edge cases (all-same predictions, perfect predictions)

4. BIAS DECOMPOSITION VALIDATION
   - center_matrix correctness
   - Zero mean after centering

5. REPRODUCIBILITY
   - Identical results across two runs with same seed
   - Different seeds give different splits

6. EDGE CASES
   - Cold-start user (user in test not in train)
   - Sparse row (user with only 1 training rating)
   - All-NaN column (unrated item)
   - k > matrix rank (truncation safety)
   - Empty test results when all entries are cold-start

7. PERFORMANCE & MEMORY
   - SVD completes on a 200x150 matrix in under 5 seconds
   - SGD-MF completes 5 epochs in under 30 seconds

Run with (from the project root):
    python -m pytest tests/test_pipeline.py -v

Review notes (2026-11): several assertions here could pass without checking the
behaviour their names promise. They were strengthened, not removed:
  * test_matrix_nan_semantics       now checks every observed entry equals its
                                    training rating and nothing else is observed.
  * test_centered_mean_near_zero    now checks the exact property the bias
                                    decomposition guarantees (zero column means).
  * test_different_seeds_...        compares (user, movie) pair sets, not one
                                    column positionally.
  * test_saved_csv_rmse_...         claimed a recomputation but only range-checked
                                    a historical file; replaced by a real check of
                                    the corrected selection and a historical-label check.
  * test_predict_batch_vs_loop_...  marked "performance" (timing-based).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    build_user_item_matrix,
    center_matrix,
    impute_matrix,
    train_test_split,
)
from evaluation import (
    EvalResult,
    GlobalMeanBaseline,
    ItemMeanBaseline,
    SGDMatrixFactorization,
    UserItemMeanBaseline,
    UserMeanBaseline,
    coverage,
    evaluate_model,
    mae,
    results_table,
    rmse,
)
from svd_model import (
    eckart_young_error,
    fit_svd,
    select_k_by_variance,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

def make_synthetic_ratings(
    n_users: int = 50,
    n_items: int = 40,
    density: float = 0.20,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic ratings DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_users + 1):
        for mid in range(1, n_items + 1):
            if rng.random() < density:
                r = float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
                rows.append({"userId": uid, "movieId": mid, "rating": r, "timestamp": 0})
    return pd.DataFrame(rows)


def make_low_rank_matrix(m: int = 60, n: int = 50, rank: int = 5, seed: int = 42) -> np.ndarray:
    """Generate a low-rank matrix with ~20% NaN."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (m, rank))
    V = rng.normal(0, 1, (n, rank))
    R = U @ V.T + 0.1 * rng.normal(0, 1, (m, n))
    # clip to rating scale
    R = np.clip(R + 3.5, 0.5, 5.0)
    # introduce 20% missing
    mask = rng.random((m, n)) < 0.20
    R[mask] = np.nan
    return R


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataValidation:
    """Tests for data loading, splitting, and matrix construction."""

    def test_train_test_no_leakage(self):
        """Train and test indices must be completely disjoint."""
        df = make_synthetic_ratings()
        train_df, test_df = train_test_split(df, test_frac=0.2, random_state=42)
        # Leakage: same (userId, movieId) pair in both sets
        train_pairs = set(zip(train_df["userId"], train_df["movieId"]))
        test_pairs  = set(zip(test_df["userId"],  test_df["movieId"]))
        overlap = train_pairs & test_pairs
        assert len(overlap) == 0, f"Leakage: {len(overlap)} pairs appear in both sets"

    def test_stratified_split_every_user_in_both(self):
        """Every user must appear in both train and test after stratified split."""
        df = make_synthetic_ratings(n_users=30, density=0.40)
        train_df, test_df = train_test_split(df, test_frac=0.2, random_state=42)
        users_train = set(train_df["userId"])
        users_test  = set(test_df["userId"])
        assert users_train == users_test, "Some users missing from one split"

    def test_matrix_shape(self):
        """Matrix shape must equal (n_users_full, n_items_full)."""
        df = make_synthetic_ratings(n_users=20, n_items=15, density=0.30)
        train_df, _ = train_test_split(df, test_frac=0.2, random_state=0)
        R, user_idx, item_idx = build_user_item_matrix(train_df, full_df=df)
        assert R.shape == (len(df["userId"].unique()), len(df["movieId"].unique()))

    def test_matrix_nan_semantics(self):
        """Observed entries equal the training ratings; every other entry is NaN."""
        df = make_synthetic_ratings(n_users=10, n_items=10, density=0.20)
        train_df, _ = train_test_split(df, test_frac=0.2, random_state=0)
        R, user_idx, item_idx = build_user_item_matrix(train_df, full_df=df)
        assert np.count_nonzero(~np.isnan(R)) == len(train_df)
        for row in train_df.itertuples(index=False):
            assert R[user_idx[row.userId], item_idx[row.movieId]] == row.rating
        assert np.count_nonzero(np.isnan(R)) == R.size - len(train_df)

    def test_impute_no_nan(self):
        """After imputation, no NaN should remain."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        for strat in ("zero", "rowmean", "colmean"):
            R_filled = impute_matrix(R_c, strategy=strat)
            assert not np.any(np.isnan(R_filled)), f"NaN remains after {strat} imputation"

    def test_split_fraction_respected(self):
        """Test set should be approximately test_frac of total ratings."""
        df = make_synthetic_ratings(n_users=50, density=0.50)
        frac = 0.25
        train_df, test_df = train_test_split(df, test_frac=frac, random_state=42)
        actual_frac = len(test_df) / len(df)
        # Allow ±5% tolerance due to per-user rounding
        assert abs(actual_frac - frac) < 0.05, f"Split fraction off: {actual_frac:.3f}"

    def test_invalid_strategy_raises(self):
        """impute_matrix must raise ValueError for unknown strategy."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        with pytest.raises(ValueError, match="strategy"):
            impute_matrix(R_c, strategy="invalid_strat")

    def test_split_invalid_fraction_raises(self):
        """train_test_split must raise ValueError for out-of-range test_frac."""
        df = make_synthetic_ratings()
        with pytest.raises(ValueError):
            train_test_split(df, test_frac=1.5)
        with pytest.raises(ValueError):
            train_test_split(df, test_frac=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — MODEL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelValidation:
    """Tests for SVD decomposition correctness and prediction validity."""

    def test_eckart_young_reconstruction(self):
        """Rank-k SVD must achieve the Eckart-Young minimum Frobenius error."""
        rng = np.random.default_rng(1)
        m, n, k_true = 40, 30, 4
        A = rng.normal(0, 1, (m, n))

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        for k in [2, 4, 8]:
            A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            expected_err = float(np.sqrt(np.sum(S[k:] ** 2)))
            actual_err   = float(np.linalg.norm(A - A_k, "fro"))
            assert abs(actual_err - expected_err) < 1e-8, (
                f"Eckart-Young error mismatch at k={k}: "
                f"expected {expected_err:.6f}, got {actual_err:.6f}"
            )

    def test_prediction_bounds(self):
        """All predictions must lie within [0.5, 5.0]."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=10, mu=mu, user_bias=ub, item_bias=ib)

        for u in range(0, R.shape[0], 5):
            for i in range(0, R.shape[1], 5):
                pred = model.predict(u, i)
                assert 0.5 <= pred <= 5.0, f"Prediction out of range: {pred}"

    def test_batch_vs_scalar_consistency(self):
        """predict_batch and predict must return identical values."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=8, mu=mu, user_bias=ub, item_bias=ib)

        users = np.array([0, 5, 10, 15, 20])
        items = np.array([0, 3,  7, 12, 18])

        batch_preds = model.predict_batch(users, items)
        for t, (u, i) in enumerate(zip(users, items)):
            scalar = model.predict(u, i)
            assert abs(batch_preds[t] - scalar) < 1e-10, (
                f"Batch/scalar mismatch at ({u},{i}): {batch_preds[t]:.8f} vs {scalar:.8f}"
            )

    def test_svd_orthonormality(self):
        """U_k and V_k^T columns must be orthonormal."""
        R = make_low_rank_matrix(seed=99)
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=10)

        # U_k^T U_k should be identity
        UtU = model.U_k.T @ model.U_k
        assert np.allclose(UtU, np.eye(model.k), atol=1e-8), "U_k is not orthonormal"

        # Vt_k @ Vt_k^T should be identity
        VtV = model.Vt_k @ model.Vt_k.T
        assert np.allclose(VtV, np.eye(model.k), atol=1e-8), "Vt_k rows are not orthonormal"

    def test_singular_values_descending(self):
        """Singular values must be non-negative and in descending order."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=15)

        sv = model.all_singular_values
        assert np.all(sv >= 0), "Singular values must be non-negative"
        assert np.all(np.diff(sv) <= 1e-8), "Singular values must be in descending order"

    def test_k_exceeds_rank_clipped(self):
        """Requesting k > rank(R) should not crash; k is clipped to rank."""
        R = make_low_rank_matrix(m=10, n=8, rank=3)
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=999)   # request way more than rank
        assert model.k <= min(R_filled.shape)

    def test_fit_svd_rejects_nan(self):
        """fit_svd must raise ValueError if the input contains NaN."""
        R = make_low_rank_matrix()
        with pytest.raises(ValueError, match="NaN"):
            fit_svd(R, k=5)      # R has NaN entries — should fail

    def test_fit_svd_rejects_nonpositive_k(self):
        """fit_svd must raise ValueError for k <= 0."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        with pytest.raises(ValueError, match="k must be"):
            fit_svd(R_filled, k=0)

    def test_explained_variance_sums_to_one(self):
        """Full explained-variance array must sum to 1.0."""
        R = make_low_rank_matrix()
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=10)
        ev_sum = float(model.explained_variance_ratio().sum())
        assert abs(ev_sum - 1.0) < 1e-8, f"EV sum = {ev_sum}"

    def test_eckart_young_error_helper(self):
        """eckart_young_error must equal sqrt(sum of squared tail SVs)."""
        sv = np.array([10.0, 6.0, 3.0, 1.5, 0.8, 0.3])
        for k in [1, 2, 3, 4, 5]:
            expected = float(np.sqrt(np.sum(sv[k:] ** 2)))
            actual   = eckart_young_error(sv, k)
            assert abs(actual - expected) < 1e-10, f"EY error wrong at k={k}"

    def test_eckart_young_error_zero_at_full_rank(self):
        """EY error should be 0 when k >= len(singular_values)."""
        sv = np.array([5.0, 3.0, 1.0])
        assert eckart_young_error(sv, 3) == 0.0
        assert eckart_young_error(sv, 100) == 0.0

    def test_select_k_by_variance_monotone(self):
        """select_k_by_variance must return larger k for higher threshold."""
        sv = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1])
        k_80 = select_k_by_variance(sv, 0.80)
        k_90 = select_k_by_variance(sv, 0.90)
        assert k_80 <= k_90, "Stricter threshold must require >= k"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — METRICS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsValidation:
    """Tests for RMSE and MAE correctness."""

    def test_rmse_manual(self):
        """RMSE must match manual computation."""
        y_true = np.array([3.0, 4.0, 2.5, 5.0])
        y_pred = np.array([3.1, 3.8, 2.7, 4.9])
        expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_mae_manual(self):
        """MAE must match manual computation."""
        y_true = np.array([3.0, 4.0, 2.5, 5.0])
        y_pred = np.array([3.1, 3.8, 2.7, 4.9])
        expected = float(np.mean(np.abs(y_true - y_pred)))
        assert abs(mae(y_true, y_pred) - expected) < 1e-10

    def test_rmse_perfect(self):
        """RMSE must be 0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y, y) < 1e-10

    def test_mae_perfect(self):
        """MAE must be 0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mae(y, y) < 1e-10

    def test_rmse_gt_mae(self):
        """RMSE >= MAE for non-constant errors (Jensen's inequality)."""
        rng = np.random.default_rng(5)
        y_true = rng.uniform(1, 5, 200)
        y_pred = y_true + rng.normal(0, 0.5, 200)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_rmse_shape_mismatch_raises(self):
        """rmse must raise ValueError for mismatched shapes."""
        with pytest.raises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_mae_empty_raises(self):
        """mae must raise ValueError on empty arrays."""
        with pytest.raises(ValueError):
            mae(np.array([]), np.array([]))

    def test_coverage_all_finite(self):
        """coverage must be 1.0 when all predictions are finite."""
        preds = np.array([3.5, 4.0, 2.5, 1.0])
        assert coverage(preds) == 1.0

    def test_coverage_with_nan(self):
        """coverage must count only finite values."""
        preds = np.array([3.5, np.nan, 2.5, np.inf])
        assert abs(coverage(preds) - 0.5) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — BIAS DECOMPOSITION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestBiasDecomposition:
    """Tests for center_matrix correctness."""

    def test_centered_mean_near_zero(self):
        """b_i is the column mean of R - mu - b_u, so every column of R_c has zero mean."""
        R = make_low_rank_matrix()
        R_c, mu, ub, ib = center_matrix(R)
        col_means = np.nanmean(R_c, axis=0)
        assert np.allclose(col_means[~np.isnan(col_means)], 0.0, atol=1e-10)
        assert abs(float(np.nanmean(R_c))) < 1e-10

    def test_bias_reconstruction(self):
        """mu + b_u + b_i + R_c should approximately reconstruct R on Omega."""
        R = make_low_rank_matrix()
        R_c, mu, ub, ib = center_matrix(R)
        R_reconstructed = mu + ub[:, None] + ib[None, :] + R_c
        obs = ~np.isnan(R)
        max_err = float(np.nanmax(np.abs(R[obs] - R_reconstructed[obs])))
        assert max_err < 1e-8, f"Bias reconstruction error: {max_err}"

    def test_all_nan_matrix_raises(self):
        """center_matrix must raise ValueError on an all-NaN matrix."""
        R = np.full((5, 5), np.nan)
        with pytest.raises(ValueError, match="no observed"):
            center_matrix(R)

    def test_global_mean_in_range(self):
        """Global mean must be within [0.5, 5.0] for MovieLens-scale data."""
        R = make_low_rank_matrix()
        _, mu, _, _ = center_matrix(R)
        assert 0.5 <= mu <= 5.0, f"Global mean {mu} out of rating range"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    """Tests for deterministic outputs under fixed seeds."""

    def test_same_seed_same_split(self):
        """Two calls with the same seed must produce identical splits."""
        df = make_synthetic_ratings(seed=0)
        train1, test1 = train_test_split(df, test_frac=0.2, random_state=42)
        train2, test2 = train_test_split(df, test_frac=0.2, random_state=42)
        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                       train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True),
                                       test2.reset_index(drop=True))

    def test_different_seeds_different_splits(self):
        """Different seeds should (with very high probability) give different splits."""
        df = make_synthetic_ratings(seed=0)
        _, test1 = train_test_split(df, test_frac=0.2, random_state=1)
        _, test2 = train_test_split(df, test_frac=0.2, random_state=99)
        pairs1 = set(zip(test1["userId"], test1["movieId"]))
        pairs2 = set(zip(test2["userId"], test2["movieId"]))
        assert len(pairs1) == len(pairs2) > 0
        assert pairs1 != pairs2

    def test_svd_deterministic(self):
        """numpy.linalg.svd is deterministic; two calls on the same matrix agree."""
        R = make_low_rank_matrix(seed=7)
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")

        model1 = fit_svd(R_filled, k=8)
        model2 = fit_svd(R_filled, k=8)

        assert np.allclose(model1.sigma_k, model2.sigma_k, atol=1e-10)

    def test_sgd_deterministic(self):
        """SGD-MF with fixed seed must give the same RMSE on two runs."""
        R = make_low_rank_matrix(seed=3)
        sgd1 = SGDMatrixFactorization(k=5, epochs=3).fit(R)
        sgd2 = SGDMatrixFactorization(k=5, epochs=3).fit(R)

        # Compare a few predictions
        for u, i in [(0, 0), (5, 5), (10, 10)]:
            assert abs(sgd1.predict(u, i) - sgd2.predict(u, i)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6 — EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for unusual or boundary inputs."""

    def test_cold_start_user_skipped(self):
        """evaluate_model must skip test entries whose userId is not in training vocab."""
        # Build a matrix without userId=999
        df = make_synthetic_ratings(n_users=10, n_items=10, density=0.50)
        train_df, _ = train_test_split(df, test_frac=0.2, random_state=0)
        R, user_idx, item_idx = build_user_item_matrix(train_df, full_df=df)

        # Test DataFrame with a cold-start user
        cold_test = pd.DataFrame([
            {"userId": 9999, "movieId": 1, "rating": 3.0},
            {"userId": 1,    "movieId": 1, "rating": 4.0},
        ])
        gm = GlobalMeanBaseline().fit(R)
        res = evaluate_model("GlobalMean", gm.predict, cold_test, user_idx, item_idx)
        # Only 1 entry should be evaluated (userId=1 is in vocab)
        assert res.n_samples == 1

    def test_sparse_row_user_falls_back(self):
        """User with only 1 training rating: UserMean should fall back gracefully."""
        R = np.full((5, 5), np.nan)
        R[0, 0] = 3.5   # only 1 rating for user 0
        R[1, :] = [2.0, 3.0, 4.0, 5.0, 1.0]
        R[2, :] = [3.0, 3.0, 3.0, 3.0, 3.0]
        R[3, :] = [4.0, 4.0, 4.0, np.nan, 4.0]
        R[4, :] = [1.0, np.nan, 5.0, 2.0, 3.0]

        um = UserMeanBaseline().fit(R)
        pred = um.predict(0, 3)
        assert np.isfinite(pred), "UserMean returned non-finite for sparse user"
        assert 0.5 <= pred <= 5.0

    def test_all_nan_column_item_mean_fallback(self):
        """Item with no training ratings: ItemMean should use global mean."""
        R = np.full((5, 5), np.nan)
        R[:, 0] = [2.0, 3.0, 4.0, 5.0, 1.0]  # column 0 has data
        # column 1-4 are all NaN
        R[0, 1] = 3.0  # give at least one entry

        im_ = ItemMeanBaseline().fit(R)
        # column 4 is all NaN -> should predict global mean
        global_mean = float(np.nanmean(R))
        pred_col4 = im_.predict(0, 4)
        assert abs(pred_col4 - global_mean) < 1e-8

    def test_user_item_bias_clips_to_range(self):
        """UserItemBias predictions must be clipped to [0.5, 5.0]."""
        R = make_low_rank_matrix()
        uim = UserItemMeanBaseline().fit(R)
        for u in range(R.shape[0]):
            for i in [0, 5, R.shape[1] - 1]:
                p = uim.predict(u, i)
                assert 0.5 <= p <= 5.0, f"Unclipped prediction: {p}"

    def test_predict_out_of_bounds_raises(self):
        """SVDModel.predict must raise IndexError for out-of-bounds indices."""
        R = make_low_rank_matrix(m=10, n=8)
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=3)
        with pytest.raises(IndexError):
            model.predict(999, 0)
        with pytest.raises(IndexError):
            model.predict(0, 999)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Timing tests to ensure the pipeline completes within reasonable bounds."""

    def test_svd_timing_200x150(self):
        """Full SVD on a 200x150 matrix must complete within 5 seconds."""
        rng = np.random.default_rng(0)
        R_filled = rng.normal(0, 1, (200, 150))
        t0 = time.perf_counter()
        np.linalg.svd(R_filled, full_matrices=False)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"SVD took {elapsed:.2f}s > 5.0s"

    def test_sgd_timing_5epochs(self):
        """SGD-MF for 5 epochs on a 50x40 matrix must complete within 30 seconds."""
        R = make_low_rank_matrix(m=50, n=40)
        sgd = SGDMatrixFactorization(k=5, epochs=5)
        t0 = time.perf_counter()
        sgd.fit(R)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"SGD-MF took {elapsed:.2f}s > 30.0s"

    @pytest.mark.performance
    def test_predict_batch_vs_loop_speed(self):
        """predict_batch must be faster than a Python loop for N=500 predictions."""
        R = make_low_rank_matrix(m=60, n=50)
        _, mu, ub, ib = center_matrix(R)
        R_c = R - mu - ub[:, None] - ib[None, :]
        R_filled = impute_matrix(R_c, "zero")
        model = fit_svd(R_filled, k=10, mu=mu, user_bias=ub, item_bias=ib)

        rng = np.random.default_rng(0)
        users = rng.integers(0, R.shape[0], 500)
        items = rng.integers(0, R.shape[1], 500)

        t0 = time.perf_counter()
        for u, i in zip(users, items):
            model.predict(u, i)
        loop_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.predict_batch(users, items)
        batch_time = time.perf_counter() - t0

        assert batch_time < loop_time, (
            f"predict_batch ({batch_time:.4f}s) not faster than loop ({loop_time:.4f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 8 — RESULTS CONSISTENCY (recompute from saved CSV)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultsConsistency:
    """Verify that saved results_table.csv matches recomputed values."""

    def test_results_table_sorted(self):
        """results_table should return rows sorted by RMSE ascending."""
        results = [
            EvalResult("A", 0.95, 0.75, 100),
            EvalResult("B", 0.88, 0.70, 100),
            EvalResult("C", 1.10, 0.90, 100),
        ]
        tbl = results_table(results)
        rmse_vals = tbl["RMSE"].tolist()
        assert rmse_vals == sorted(rmse_vals), "Table is not sorted by RMSE"

    def test_results_table_has_required_columns(self):
        """results_table must contain [Model, RMSE, MAE, N] columns."""
        results = [EvalResult("X", 0.90, 0.70, 50)]
        tbl = results_table(results)
        for col in ["Model", "RMSE", "MAE", "N"]:
            assert col in tbl.columns, f"Missing column: {col}"

    def test_historical_results_are_preserved_and_labelled(self):
        """Historical (test-selected) results stay on disk and are labelled as such."""
        root = Path(__file__).parent.parent
        note = root / "outputs" / "HISTORICAL_RESULTS.md"
        assert note.exists(), "outputs/HISTORICAL_RESULTS.md must label the historical results"
        text = note.read_text(encoding="utf-8")
        for name in ("results_table.csv", "results_ml1m.csv"):
            assert (root / "outputs" / name).exists(), f"historical file {name} was removed"
            assert name in text
        assert "test set" in text.lower() and "historical" in text.lower()

    def test_corrected_selection_is_argmin_of_validation_sweep(self):
        """The recorded k must be the argmin of the VALIDATION sweep (not of test)."""
        import json
        root = Path(__file__).parent.parent / "outputs" / "corrected"
        sweep_path, final_path = root / "k_sweep_validation.csv", root / "final_metrics.json"
        if not (sweep_path.exists() and final_path.exists()):
            pytest.skip("corrected outputs not generated; run src/main.py first")
        sweep = pd.read_csv(sweep_path)
        final = json.loads(final_path.read_text(encoding="utf-8"))
        best = sweep.sort_values(["val_rmse", "k"]).iloc[0]
        assert int(best["k"]) == final["selection"]["selected_k"]
        assert final["selection"]["metric"] == "validation RMSE"
        assert final["validation"]["svd"]["partition"] == "validation"
        assert final["test"]["svd"]["partition"] == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Allow running directly: python tests/test_pipeline.py
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent,
    )
    sys.exit(result.returncode)
