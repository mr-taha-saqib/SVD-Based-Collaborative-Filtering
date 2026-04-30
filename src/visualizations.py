"""
visualizations.py
=================
Publication-quality plots for the SVD recommendation system project.

All save functions accept an optional `plots_dir` argument so the caller
(main.py) can pass the directory from config.yaml.

Figures:
    01 singular_value_decay     - scree plot of sigma_l vs l
    02 explained_variance       - cumulative EV(k) vs k
    03 rmse_vs_k                - test RMSE + EY bound vs k
    04 mae_vs_k                 - test MAE vs k
    05 rating_distribution      - histogram of observed ratings
    06 sparsity_pattern         - heatmap of observed entries
    07 reconstruction_heatmap   - original vs rank-k reconstruction block
    08 baseline_comparison      - horizontal bar chart of all model RMSEs/MAEs
    09 latent_factors           - 2-D PCA projection of user/item embeddings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless backend — no GUI required

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
})


def _save(fig: plt.Figure, name: str, plots_dir: Path) -> Path:
    """Save figure and close it.  Returns the saved path."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path)
    return path


# ── 01. Singular value decay ──────────────────────────────────────────────────
def plot_singular_value_decay(
    singular_values: np.ndarray,
    top_n: int = 100,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Scree plot: sigma_l vs index l.

    Rapid decay confirms low intrinsic rank in the rating matrix.
    Reference lines at k=10 and k=50 mark typical optimal ranks.

    Args:
        singular_values: Full singular-value array, descending.
        top_n:           How many singular values to display.
        plots_dir:       Output directory.  Defaults to outputs/plots/.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    sv = singular_values[:top_n]
    idx = np.arange(1, len(sv) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, sv, "b-o", markersize=3, linewidth=1.5, label="Singular values")
    ax.set_xlabel("Index $l$")
    ax.set_ylabel("Singular value $\\sigma_l$")
    ax.set_title("Singular Value Spectrum (Scree Plot)")
    ax.set_xlim(0, len(sv) + 1)

    for kv, col, ls in [(10, "red", "--"), (50, "green", "-.")]:
        if kv <= len(sv):
            ax.axvline(x=kv, color=col, linestyle=ls, alpha=0.8, label=f"k={kv}")

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "01_singular_value_decay", plots_dir)


# ── 02. Explained variance ────────────────────────────────────────────────────
def plot_explained_variance(
    singular_values: np.ndarray,
    ks: Optional[list[int]] = None,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Cumulative explained variance ratio EV(k) vs rank k.

    EV(k) = sum_{l=1}^{k} sigma_l^2 / sum_l sigma_l^2

    Horizontal dashed line at 90% threshold.

    Args:
        singular_values: Full singular-value array.
        ks:              k-values to annotate with vertical lines.
        plots_dir:       Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    if ks is None:
        ks = [5, 10, 20, 50, 100]

    sq  = singular_values ** 2
    cum = np.cumsum(sq) / sq.sum()
    max_k = min(300, len(cum))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum * 100, "b-", linewidth=2,
            label="Cumulative EV")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(y=90, color="gray", linestyle=":", alpha=0.8, label="90% threshold")
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance vs Rank $k$")
    ax.set_xlim(0, max_k)
    ax.set_ylim(0, 101)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ks)))
    for k, c in zip(ks, colors):
        if k <= len(cum):
            ax.axvline(x=k, color=c, linestyle="--", alpha=0.75, label=f"k={k}")
            ax.scatter([k], [cum[k - 1] * 100], color=c, zorder=5, s=60)

    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "02_explained_variance", plots_dir)


# ── 03. RMSE vs k ─────────────────────────────────────────────────────────────
def plot_rmse_vs_k(
    k_values:   list[int],
    rmse_values: list[float],
    ey_bounds:   Optional[list[float]] = None,
    plots_dir:   Optional[Path] = None,
) -> Path:
    """Test RMSE and Eckart-Young theoretical bound vs rank k.

    The gap between the two curves equals Delta(k) from the report:
        Delta(k) = RMSE_test(k) - (1/sqrt(|Omega|)) * ||R - R_k||_F

    Args:
        k_values:    List of k values used in the sweep.
        rmse_values: Corresponding test RMSE values.
        ey_bounds:   Normalised Eckart-Young lower bounds (optional).
        plots_dir:   Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_values, rmse_values, "b-o", linewidth=2, markersize=6,
            label="Test RMSE")

    if ey_bounds:
        ax.plot(k_values, ey_bounds, "r--^", linewidth=1.5, markersize=5,
                label="EY lower bound (normalised)")

    best_k    = k_values[int(np.argmin(rmse_values))]
    best_rmse = float(min(rmse_values))
    ax.axvline(x=best_k, color="green", linestyle=":", alpha=0.8,
               label=f"Best k={best_k}  (RMSE={best_rmse:.4f})")

    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("RMSE")
    ax.set_title("Test RMSE vs Rank $k$  (U-shaped overfitting curve)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "03_rmse_vs_k", plots_dir)


# ── 04. MAE vs k ──────────────────────────────────────────────────────────────
def plot_mae_vs_k(
    k_values:   list[int],
    mae_values: list[float],
    plots_dir:  Optional[Path] = None,
) -> Path:
    """Test MAE vs rank k.

    Args:
        k_values:   List of k values.
        mae_values: Corresponding test MAE values.
        plots_dir:  Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_values, mae_values, "g-s", linewidth=2, markersize=6)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("MAE")
    ax.set_title("Test MAE vs Rank $k$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "04_mae_vs_k", plots_dir)


# ── 05. Rating distribution ───────────────────────────────────────────────────
def plot_rating_distribution(
    df: pd.DataFrame,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Histogram of all observed ratings.

    Args:
        df:        DataFrame with a "rating" column.
        plots_dir: Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    bins = [x - 0.25 for x in np.arange(0.5, 5.6, 0.5)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["rating"], bins=bins, edgecolor="black", color="steelblue", alpha=0.85)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Observed Ratings (MovieLens)")
    ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "05_rating_distribution", plots_dir)


# ── 06. Sparsity pattern ──────────────────────────────────────────────────────
def plot_sparsity(
    R: np.ndarray,
    max_dim: int = 200,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Heatmap of observed/missing pattern in R.

    Sub-samples to max_dim x max_dim for display performance.

    Args:
        R:         Rating matrix with NaN for unobserved entries.
        max_dim:   Maximum display dimension (rows and columns).
        plots_dir: Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    step_r = max(1, R.shape[0] // max_dim)
    step_c = max(1, R.shape[1] // max_dim)
    R_sub  = R[::step_r, ::step_c]
    mask   = (~np.isnan(R_sub)).astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(mask, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Observed Ratings Pattern  (blue = observed, white = missing)")
    ax.set_xlabel("Item index (sampled)")
    ax.set_ylabel("User index (sampled)")
    density = mask.mean()
    ax.set_xlabel(f"Item index (sampled)  |  density = {density:.2%}")
    fig.tight_layout()
    return _save(fig, "06_sparsity_pattern", plots_dir)


# ── 07. Reconstruction heatmap ────────────────────────────────────────────────
def plot_reconstruction_heatmap(
    R_original:     np.ndarray,
    R_reconstructed: np.ndarray,
    block: int = 50,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Side-by-side heatmap: original R vs rank-k reconstruction.

    Args:
        R_original:      Original rating matrix (with NaN for missing).
        R_reconstructed: Rank-k reconstruction from SVDModel.reconstruct().
        block:           Size of the display block (top-left corner).
        plots_dir:       Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    orig  = R_original[:block, :block]
    recon = R_reconstructed[:block, :block]
    cmap  = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    orig_masked = np.ma.masked_invalid(orig)
    im0 = axes[0].imshow(orig_masked, aspect="auto", cmap=cmap, vmin=0.5, vmax=5.0,
                          interpolation="nearest")
    axes[0].set_title("Original $R$  (grey = missing)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon, aspect="auto", cmap=cmap, vmin=0.5, vmax=5.0,
                          interpolation="nearest")
    axes[1].set_title("Rank-$k$ Reconstruction $\\hat{R}$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Item index")
        ax.set_ylabel("User index")

    fig.suptitle(f"Reconstruction comparison (first {block}x{block} block)", y=1.01)
    fig.tight_layout()
    return _save(fig, "07_reconstruction_heatmap", plots_dir)


# ── 08. Baseline comparison ───────────────────────────────────────────────────
def plot_baseline_comparison(
    results_df: pd.DataFrame,
    plots_dir: Optional[Path] = None,
) -> Path:
    """Horizontal bar chart comparing RMSE and MAE across all models.

    Args:
        results_df: DataFrame with columns [Model, RMSE, MAE, N].
        plots_dir:  Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    n_models = len(results_df)
    colors   = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, n_models * 0.65 + 1)))
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        df_s = results_df.sort_values(metric, ascending=False)
        bars = ax.barh(df_s["Model"], df_s[metric], color=colors[:len(df_s)],
                       edgecolor="black", alpha=0.85)
        ax.set_xlabel(metric)
        ax.set_title(f"Model Comparison -- {metric}")
        ax.grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, df_s[metric]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    return _save(fig, "08_baseline_comparison", plots_dir)


# ── 09. Latent factor 2-D projection ─────────────────────────────────────────
def plot_latent_factors(
    U_k:     np.ndarray,
    Vt_k:    np.ndarray,
    n_users: int = 200,
    n_items: int = 200,
    plots_dir: Optional[Path] = None,
) -> Path:
    """2-D PCA projection of user and item latent vectors.

    Projects high-dimensional (k-dimensional) embeddings to 2-D via PCA,
    giving a geometric view of the learned latent space.  Clusters in the
    plot correspond to groups of users/items with similar latent profiles.

    Args:
        U_k:      Left singular vectors, shape (m, k).
        Vt_k:     Right singular vectors transposed, shape (k, n).
        n_users:  Max number of users to display.
        n_items:  Max number of items to display.
        plots_dir: Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn not available; skipping latent factor plot.")
        return plots_dir / "09_latent_factors_skipped.png"

    U_sub = U_k[:min(n_users, U_k.shape[0])]
    V_sub = Vt_k.T[:min(n_items, Vt_k.shape[1])]

    all_pts = np.vstack([U_sub, V_sub])
    pca     = PCA(n_components=2)
    pts_2d  = pca.fit_transform(all_pts)

    U_2d = pts_2d[:len(U_sub)]
    V_2d = pts_2d[len(U_sub):]
    var  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(U_2d[:, 0], U_2d[:, 1], s=15, alpha=0.5, c="royalblue",
               label=f"Users (n={len(U_sub)})")
    ax.scatter(V_2d[:, 0], V_2d[:, 1], s=15, alpha=0.5, c="tomato",
               label=f"Items (n={len(V_sub)})")
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)")
    ax.set_title("Latent Factor Space  (PCA projection to 2-D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "09_latent_factors", plots_dir)


# ── 10. Cross-dataset comparison ──────────────────────────────────────────────
def plot_crossdataset_comparison(
    small_ks:        list[int],
    small_rmse:      list[float],
    m1_ks:           list[int],
    m1_rmse:         list[float],
    small_baselines: pd.DataFrame,
    m1_baselines:    pd.DataFrame,
    plots_dir:       Optional[Path] = None,
) -> Path:
    """Two-panel figure comparing ML-small and ML-1M results.

    Panel 1: RMSE vs k for ML-1M k-sweep; best ML-small point marked as a
             horizontal reference line (full k-sweep data often unavailable).
    Panel 2: Percentage RMSE reduction over Global Mean for each model tier,
             both datasets side-by-side.

    Args:
        small_ks / m1_ks:           k-sweep values for each dataset.
        small_rmse / m1_rmse:       Test RMSE lists for each dataset.
        small_baselines / m1_baselines: results_table DataFrames (Model, RMSE).
        plots_dir:                  Output directory.

    Returns:
        Path to the saved PNG.
    """
    plots_dir = plots_dir or _DEFAULT_PLOTS_DIR

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: RMSE vs k ────────────────────────────────────────────────────
    ax = axes[0]

    best_m1    = float(min(m1_rmse))
    bk_m1      = m1_ks[int(np.argmin(m1_rmse))]

    ax.plot(m1_ks, m1_rmse, "r-s", markersize=6, linewidth=2,
            label="ML-1M  (6040u x 3706i)")
    ax.axvline(x=bk_m1, color="red", linestyle=":", alpha=0.7,
               label=f"Optimal k={bk_m1} (RMSE={best_m1:.4f})")

    if len(small_rmse) > 1:
        best_small = float(min(small_rmse))
        bk_small   = small_ks[int(np.argmin(small_rmse))]
        ax.plot(small_ks, small_rmse, "b-o", markersize=6, linewidth=2,
                label="ML-small  (610u x 9724i)")
        ax.axvline(x=bk_small, color="blue", linestyle=":", alpha=0.7,
                   label=f"Optimal k={bk_small} (RMSE={best_small:.4f})")
    else:
        best_small = float(small_rmse[0])
        bk_small   = small_ks[0]
        ax.axhline(y=best_small, color="blue", linestyle="--", linewidth=2, alpha=0.8,
                   label=f"ML-small best  k={bk_small} (RMSE={best_small:.4f})")
        ax.scatter([bk_small], [best_small], color="blue", s=80, zorder=5)

    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Test RMSE")
    ax.set_title("RMSE vs Rank $k$: Cross-Dataset Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: % improvement over Global Mean ───────────────────────────────
    ax2 = axes[1]
    model_order = ["Item Mean", "User Mean", "User+Item Bias", "SVD"]

    def _pct_improvement(df: pd.DataFrame) -> dict[str, float]:
        gm = df.loc[df["Model"].str.contains("Global Mean"), "RMSE"].values
        if len(gm) == 0:
            return {}
        gm_rmse = gm[0]
        out = {}
        for m in model_order:
            row = df[df["Model"].str.contains(m, regex=False)]
            if len(row):
                out[m] = (gm_rmse - row["RMSE"].values[0]) / gm_rmse * 100
        return out

    pct_small = _pct_improvement(small_baselines)
    pct_m1    = _pct_improvement(m1_baselines)

    labels = [m for m in model_order if m in pct_small and m in pct_m1]
    x      = np.arange(len(labels))
    w      = 0.35

    bars1 = ax2.bar(x - w/2, [pct_small[l] for l in labels], w,
                    label="ML-small", color="steelblue",  alpha=0.85, edgecolor="black")
    bars2 = ax2.bar(x + w/2, [pct_m1[l]    for l in labels], w,
                    label="ML-1M",    color="tomato",     alpha=0.85, edgecolor="black")

    for bar in list(bars1) + list(bars2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE reduction vs Global Mean (%)")
    ax2.set_title("Generalisation: Model Hierarchy on Both Datasets")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Validation: ML-small vs ML-1M", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "10_crossdataset_comparison", plots_dir)
