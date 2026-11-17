"""
svd_model.py
============
Truncated SVD for collaborative filtering.

Optimisation problem (Eq. opt in report)
-----------------------------------------
Given R in R^(m x n) with observed set Omega, we solve:

    min_{rank(M) <= k}  ||P_Omega(R - M)||_F^2          ... (*)

where:
    P_Omega : projection onto observed entries
              (P_Omega(A))_{ij} = A_{ij} if (i,j) in Omega, else 0
    ||.||_F : Frobenius norm:  ||A||_F^2 = sum_{i,j} A_{ij}^2

For the *fully observed* case (Omega = all entries), the Eckart-Young theorem
guarantees the exact, closed-form solution:

    M* = U_k Sigma_k V_k^T                ... (**)

where (U_k, Sigma_k, V_k) are the top-k left singular vectors, singular
values, and right singular vectors of R = U Sigma V^T.

Algorithm: Eckart-Young => Practical Implementation
----------------------------------------------------
Step 1  Centre R:     R_c = R - mu - b_u[:, None] - b_i[None, :]
Step 2  Impute NaN:   R_tilde[i,j] = R_c[i,j] if observed, else 0
Step 3  Full SVD:     R_tilde = U Sigma V^T   (numpy.linalg.svd)
Step 4  Truncate:     keep top-k triplets  -> U_k, sigma_k, Vt_k
Step 5  Predict:      r_hat_ui = mu + b_u + b_i + U_k[u] . (sigma_k * Vt_k[:, i])

This is the exact Eckart-Young optimum for R_tilde, which is the best rank-k
approximation of the imputed matrix in the Frobenius norm sense.

Prediction formula (Eq. pred in report)
-----------------------------------------
    r_hat_{ui} = mu + b_u + b_i + sum_{l=1}^{k} sigma_l * U_k[u,l] * V_k[i,l]

Implemented in SVDModel.predict() and SVDModel.predict_batch().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── SVDModel dataclass ────────────────────────────────────────────────────────
@dataclass
class SVDModel:
    """Container for a fitted truncated-SVD recommendation model.

    Attributes:
        U_k:                 Left singular vectors, shape (m, k).
        sigma_k:             Top-k singular values,  shape (k,).
        Vt_k:                Right singular vectors transposed, shape (k, n).
        mu:                  Global mean rating (scalar, for de-centering).
        user_bias:           Per-user additive bias, shape (m,).
        item_bias:           Per-item additive bias, shape (n,).
        all_singular_values: Full singular-value spectrum for scree plots.
        k:                   Rank of the approximation.
    """

    U_k:                 np.ndarray
    sigma_k:             np.ndarray
    Vt_k:                np.ndarray
    mu:                  float       = 0.0
    user_bias:           np.ndarray  = field(default_factory=lambda: np.array([]))
    item_bias:           np.ndarray  = field(default_factory=lambda: np.array([]))
    all_singular_values: np.ndarray  = field(default_factory=lambda: np.array([]))
    k:                   int         = 0

    # ── single-pair prediction ─────────────────────────────────────────────────
    def predict(
        self,
        user_idx: int,
        item_idx: int,
        clip: tuple[float, float] = (0.5, 5.0),
    ) -> float:
        """Predict the rating for a single (user, item) pair.

        Implements Eq. (pred) from report:
            r_hat_{ui} = mu + b_u + b_i
                         + U_k[u] . diag(sigma_k) . Vt_k[:, i]
                       = mu + b_u + b_i
                         + sum_{l=1}^{k} sigma_l * U_k[u,l] * Vt_k[l, i]

        Code mapping:
            latent = np.dot(U_k[u] * sigma_k,  Vt_k[:, i])
            This is mathematically identical to U_k[u] @ diag(sigma_k) @ Vt_k[:, i]
            but avoids constructing the k x k diagonal matrix.

        Args:
            user_idx: Row index i of the user in the rating matrix.
            item_idx: Column index j of the item in the rating matrix.
            clip:     (min, max) clamp for valid rating range.

        Returns:
            Predicted rating, clipped to [clip[0], clip[1]].
        """
        if not (0 <= user_idx < self.U_k.shape[0]):
            raise IndexError(f"user_idx {user_idx} out of range [0, {self.U_k.shape[0]})")
        if not (0 <= item_idx < self.Vt_k.shape[1]):
            raise IndexError(f"item_idx {item_idx} out of range [0, {self.Vt_k.shape[1]})")

        # latent interaction: dot( U_k[u] * sigma, Vt[:, i] )
        latent = float(np.dot(self.U_k[user_idx] * self.sigma_k,
                              self.Vt_k[:, item_idx]))
        b_u = float(self.user_bias[user_idx]) if len(self.user_bias) else 0.0
        b_i = float(self.item_bias[item_idx]) if len(self.item_bias) else 0.0

        return float(np.clip(self.mu + b_u + b_i + latent, clip[0], clip[1]))

    # ── vectorised batch prediction ────────────────────────────────────────────
    def predict_batch(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        clip: tuple[float, float] = (0.5, 5.0),
    ) -> np.ndarray:
        """Vectorised prediction for a batch of (user, item) pairs.

        Implements the same formula as predict() for N pairs simultaneously:

            latent[t] = (U_k[u_t] * sigma_k) . Vt_k[:, i_t]

        Code mapping (see Eq. pred):
            U_scaled = U_k[users] * sigma_k[None, :]       shape (N, k)
            V_cols   = Vt_k[:, items]                       shape (k, N)
            latent   = einsum("nk,kn->n", U_scaled, V_cols) shape (N,)

        This is equivalent to taking the diagonal of U_scaled @ V_cols, which
        is the rank-k reconstruction matrix evaluated at the (u_t, i_t) entries.

        Args:
            user_indices: Array of user row indices, shape (N,).
            item_indices: Array of item column indices, shape (N,).
            clip:         (min, max) clamp for valid rating range.

        Returns:
            Predicted ratings array of shape (N,).

        Raises:
            ValueError: If user_indices and item_indices have different lengths.
        """
        user_indices = np.asarray(user_indices)
        item_indices = np.asarray(item_indices)
        if user_indices.shape != item_indices.shape:
            raise ValueError(
                f"user_indices and item_indices must have the same shape: "
                f"{user_indices.shape} vs {item_indices.shape}"
            )

        # (N, k) scaled user factors
        U_scaled = self.U_k[user_indices] * self.sigma_k[np.newaxis, :]
        # (k, N) item factor columns
        V_cols   = self.Vt_k[:, item_indices]
        # element-wise: sum over k dimension -> (N,)
        latent   = np.einsum("nk,kn->n", U_scaled, V_cols)

        b_u = self.user_bias[user_indices] if len(self.user_bias) else 0.0
        b_i = self.item_bias[item_indices] if len(self.item_bias) else 0.0

        return np.clip(self.mu + b_u + b_i + latent, clip[0], clip[1])

    # ── unseen user / item fallback ────────────────────────────────────────────
    def fallback_batch(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        clip: tuple[float, float] = (0.5, 5.0),
    ) -> np.ndarray:
        """Prediction when user and/or item is not in the training vocabulary.

        An index of -1 marks "unknown". The latent term needs both factors, so it
        is dropped: r_hat = mu + b_u (if known) + b_i (if known), clipped.
        """
        users = np.asarray(user_indices)
        items = np.asarray(item_indices)
        out = np.full(len(users), float(self.mu))
        ku, ki = users >= 0, items >= 0
        if len(self.user_bias):
            out[ku] += self.user_bias[users[ku]]
        if len(self.item_bias):
            out[ki] += self.item_bias[items[ki]]
        return np.clip(out, clip[0], clip[1])

    # ── scoring for ranking ────────────────────────────────────────────────────
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Unclipped predicted scores for every item, for the given users.

        Ranking uses unclipped scores: clipping at 5.0 would create artificial
        ties among the top items.
        """
        users = np.asarray(user_indices)
        latent = (self.U_k[users] * self.sigma_k[np.newaxis, :]) @ self.Vt_k
        out = latent + self.mu
        if len(self.user_bias):
            out = out + self.user_bias[users][:, np.newaxis]
        if len(self.item_bias):
            out = out + self.item_bias[np.newaxis, :]
        return out

    # ── persistence (no pickle) ────────────────────────────────────────────────
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Plain numeric arrays, suitable for ``np.savez`` with allow_pickle=False."""
        return {
            "U_k": self.U_k, "sigma_k": self.sigma_k, "Vt_k": self.Vt_k,
            "mu": np.array(self.mu, dtype=np.float64),
            "user_bias": np.asarray(self.user_bias, dtype=np.float64),
            "item_bias": np.asarray(self.item_bias, dtype=np.float64),
            "all_singular_values": np.asarray(self.all_singular_values, dtype=np.float64),
            "k": np.array(self.k, dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays) -> SVDModel:
        """Inverse of :meth:`to_arrays` (accepts a dict or an ``NpzFile``)."""
        return cls(
            U_k=np.asarray(arrays["U_k"]), sigma_k=np.asarray(arrays["sigma_k"]),
            Vt_k=np.asarray(arrays["Vt_k"]), mu=float(arrays["mu"]),
            user_bias=np.asarray(arrays["user_bias"]),
            item_bias=np.asarray(arrays["item_bias"]),
            all_singular_values=np.asarray(arrays["all_singular_values"]),
            k=int(arrays["k"]),
        )

    # ── full reconstruction ────────────────────────────────────────────────────
    def reconstruct(self) -> np.ndarray:
        """Return the full (m x n) rank-k reconstruction matrix.

        R_hat = mu + b_u[:, None] + b_i[None, :] + U_k @ diag(sigma_k) @ Vt_k

        This is the de-centred version of the Eckart-Young approximation.
        Useful for visualisation (heatmaps) and computing the reconstruction
        Frobenius error on observed entries.

        Returns:
            Reconstruction matrix of shape (m, n), clipped to [0.5, 5.0].
        """
        R_hat = self.U_k @ np.diag(self.sigma_k) @ self.Vt_k
        R_hat = R_hat + self.mu
        if len(self.user_bias):
            R_hat = R_hat + self.user_bias[:, np.newaxis]
        if len(self.item_bias):
            R_hat = R_hat + self.item_bias[np.newaxis, :]
        return np.clip(R_hat, 0.5, 5.0)

    # ── explained variance ─────────────────────────────────────────────────────
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each singular value.

        Eq. (EV) from report:
            ev_l = sigma_l^2 / sum_l sigma_l^2

        Uses the full spectrum (all_singular_values) so ratios are relative
        to the total variance in the imputed matrix, not just the top-k part.

        Returns:
            Array of shape (r,) with per-component explained variance fractions.
        """
        sv = self.all_singular_values
        sq = sv ** 2
        return sq / sq.sum()

    def cumulative_explained_variance(self) -> np.ndarray:
        """Cumulative sum of explained_variance_ratio.

        Returns:
            Array of shape (r,); entry k-1 gives EV(k) from the report.
        """
        return np.cumsum(self.explained_variance_ratio())


# ── fitting ────────────────────────────────────────────────────────────────────
def fit_svd(
    R_filled: np.ndarray,
    k: int,
    mu: float = 0.0,
    user_bias: Optional[np.ndarray] = None,
    item_bias: Optional[np.ndarray] = None,
) -> SVDModel:
    """Compute the truncated SVD of the imputed rating matrix.

    Implements Steps 3-4 of the algorithm described in the module docstring.

    Algorithm:
        1. Full economy SVD:  R_filled = U Sigma V^T
           via numpy.linalg.svd(full_matrices=False)
           => LAPACK dgesdd  (divide-and-conquer), complexity O(m*n*min(m,n))
        2. Truncation to rank k:
           U_k  = U[:, :k]        in R^{m x k}
           s_k  = Sigma[:k]       in R^k          (singular values)
           Vt_k = V^T[:k, :]      in R^{k x n}
        3. Eckart-Young guarantee:
           ||R_filled - U_k Sigma_k Vt_k||_F^2  is MINIMISED over all
           rank-k matrices (Theorem 1 in report).

    Args:
        R_filled:  Imputed dense matrix in R^{m x n} (no NaN entries).
        k:         Rank of approximation.
        mu:        Global mean (for de-centering predictions).
        user_bias: Per-user bias vector, shape (m,).  None => zeros.
        item_bias: Per-item bias vector, shape (n,).  None => zeros.

    Returns:
        SVDModel with the fitted decomposition and bias terms.

    Raises:
        ValueError: If k <= 0 or if R_filled contains NaN.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if np.any(np.isnan(R_filled)):
        raise ValueError("R_filled must not contain NaN — call impute_matrix() first.")

    t0 = time.perf_counter()
    logger.info("Computing full SVD of %s matrix (k=%d)...", R_filled.shape, k)

    # economy SVD: U in (m, r), S in (r,), Vt in (r, n)  where r = min(m, n)
    U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)

    r_actual = len(S)
    k = min(k, r_actual)

    U_k  = U[:, :k]
    S_k  = S[:k]
    Vt_k = Vt[:k, :]

    # Eckart-Young residual: ||R - R_k||_F  (theoretical lower bound)
    ey_error      = float(np.sqrt(np.sum(S[k:] ** 2)))
    explained_var = float(np.sum(S_k ** 2) / np.sum(S ** 2)) * 100.0
    elapsed       = time.perf_counter() - t0

    logger.info(
        "SVD done in %.2fs | k=%d | s_1=%.2f | s_k=%.4f | "
        "Explained var: %.2f%% | EY error ||R-R_k||_F=%.4f",
        elapsed, k, S[0], S_k[-1], explained_var, ey_error,
    )

    return SVDModel(
        U_k=U_k,
        sigma_k=S_k,
        Vt_k=Vt_k,
        mu=mu,
        user_bias=user_bias if user_bias is not None else np.array([]),
        item_bias=item_bias if item_bias is not None else np.array([]),
        all_singular_values=S,
        k=k,
    )


# ── k selection helpers ────────────────────────────────────────────────────────
def select_k_by_variance(
    singular_values: np.ndarray,
    threshold: float = 0.90,
) -> int:
    """Return the smallest k satisfying the explained-variance criterion.

    Eq. (k_star) from report:
        k* = min { k : sum_{l=1}^{k} sigma_l^2 / sum_l sigma_l^2 >= threshold }

    Args:
        singular_values: Full singular-value array, descending order.
        threshold:       Minimum cumulative explained variance (default 0.90).

    Returns:
        Optimal k by the variance criterion.

    Raises:
        ValueError: If threshold is not in (0, 1].
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    sq   = singular_values ** 2
    cum  = np.cumsum(sq) / sq.sum()
    hits = np.where(cum >= threshold)[0]
    k_star = int(hits[0]) + 1 if len(hits) else len(singular_values)
    logger.info(
        "Variance criterion (tau=%.0f%%): k* = %d  (EV=%.2f%%)",
        threshold * 100, k_star, cum[k_star - 1] * 100,
    )
    return k_star


def eckart_young_error(singular_values: np.ndarray, k: int) -> float:
    """Compute the Eckart-Young lower bound on the Frobenius reconstruction error.

    Eq. (ey_bound) from report:
        E(k) = ||R - R_k||_F  = sqrt( sum_{l=k+1}^{r} sigma_l^2 )

    This is the MINIMUM possible Frobenius distance between R and any rank-k
    matrix.  It serves as a theoretical lower bound on prediction error.

    Args:
        singular_values: Full singular-value array (descending).
        k:               Rank of approximation.

    Returns:
        Eckart-Young lower bound as a scalar float.
    """
    if k >= len(singular_values):
        return 0.0
    return float(np.sqrt(np.sum(singular_values[k:] ** 2)))
