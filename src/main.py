"""
main.py
=======
End-to-end pipeline driver for the SVD-based recommendation system.

All configurable values are read from config.yaml (project root).

Pipeline stages:
    1.  Setup logging
    2.  Load configuration
    3.  Download and load MovieLens ratings
    4.  Stratified train/test split
    5.  Build user-item matrix R  in R^(m x n)
    6.  Mean-centre R (bias decomposition)
    7.  Zero-impute NaN entries -> R_tilde
    8.  Compute full SVD of R_tilde  (single SVD call, reused for all k)
    9.  k-sweep: evaluate test RMSE/MAE for each k in config
    10. Auto-select best k by test RMSE
    11. Fit all baseline models
    12. Evaluate all models on the test set
    13. Print and save the results table
    14. Generate all visualisation plots
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ensure src/ is importable when running as  python src/main.py
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_ratings,
    train_test_split,
    build_user_item_matrix,
    center_matrix,
    impute_matrix,
)
from svd_model import (
    SVDModel,
    fit_svd,
    select_k_by_variance,
    eckart_young_error,
)
from evaluation import (
    evaluate_model,
    results_table,
    GlobalMeanBaseline,
    UserMeanBaseline,
    ItemMeanBaseline,
    UserItemMeanBaseline,
    SGDMatrixFactorization,
)
from visualizations import (
    plot_singular_value_decay,
    plot_explained_variance,
    plot_rmse_vs_k,
    plot_mae_vs_k,
    plot_rating_distribution,
    plot_sparsity,
    plot_reconstruction_heatmap,
    plot_baseline_comparison,
    plot_latent_factors,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_config(config_path: Path) -> dict:
    """Load YAML config; return an empty dict if the file is absent."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _setup_logging(log_file: Path, level: str = "INFO") -> None:
    """Configure root logger to write to both console and a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ── main pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Config and logging ─────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg         = _load_config(config_path)

    out_cfg  = cfg.get("output", {})
    log_file = Path(__file__).parent.parent / out_cfg.get("log_file", "outputs/run.log")
    _setup_logging(log_file, out_cfg.get("log_level", "INFO"))

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  SVD-Based Collaborative Filtering -- Full Pipeline")
    logger.info("=" * 60)

    # ── read config values with safe defaults ─────────────────────────────────
    split_cfg  = cfg.get("split",        {})
    model_cfg  = cfg.get("model",        {})
    sgd_cfg    = cfg.get("sgd_mf",       {})
    pre_cfg    = cfg.get("preprocessing",{})

    TEST_FRAC    = float(split_cfg.get("test_fraction",  0.20))
    RANDOM_STATE = int(  split_cfg.get("random_state",   42))
    K_VALUES     = list( model_cfg.get("k_values",       [5,10,20,50,100,150,200]))
    VAR_THRESH   = float(model_cfg.get("variance_threshold", 0.90))
    RATING_CLIP  = tuple(model_cfg.get("rating_clip",    [0.5, 5.0]))
    IMPUTE_STRAT = str(  pre_cfg.get("imputation_strategy","zero"))
    RUN_SGD      = bool( sgd_cfg.get("enabled",          True))
    SGD_EPOCHS   = int(  sgd_cfg.get("epochs",           20))
    SGD_LR       = float(sgd_cfg.get("learning_rate",    0.005))
    SGD_REG      = float(sgd_cfg.get("regularization",   0.02))
    SGD_LOG_INT  = int(  sgd_cfg.get("log_interval",     5))

    results_csv = Path(__file__).parent.parent / out_cfg.get("results_csv","outputs/results_table.csv")
    plots_dir   = Path(__file__).parent.parent / out_cfg.get("plots_dir",  "outputs/plots")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    df = load_ratings()

    # ── 3. Split ──────────────────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_frac=TEST_FRAC, random_state=RANDOM_STATE)

    # ── 4. Build rating matrix ────────────────────────────────────────────────
    R, user_idx, item_idx = build_user_item_matrix(train_df, full_df=df)

    # ── 5. Centre + impute ────────────────────────────────────────────────────
    R_c, mu, user_bias, item_bias = center_matrix(R)
    R_filled = impute_matrix(R_c, strategy=IMPUTE_STRAT)

    # ── 6. Full SVD (compute once, truncate repeatedly) ───────────────────────
    logger.info("Computing full economy SVD of %s matrix...", R_filled.shape)
    t_svd = time.perf_counter()
    U_full, S_full, Vt_full = np.linalg.svd(R_filled, full_matrices=False)
    logger.info("Full SVD completed in %.2fs | rank=%d | s_1=%.2f | s_min=%.6f",
                time.perf_counter() - t_svd, len(S_full), S_full[0], S_full[-1])

    # ── 7. k-sweep ────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Rank-k sweep: %s", K_VALUES)
    logger.info("-" * 50)

    rmse_list: list[float] = []
    mae_list:  list[float] = []
    ey_list:   list[float] = []
    m_dim, n_dim = R_filled.shape

    for k in K_VALUES:
        k_eff = min(k, len(S_full))
        model_k = SVDModel(
            U_k     = U_full[:, :k_eff],
            sigma_k = S_full[:k_eff],
            Vt_k    = Vt_full[:k_eff, :],
            mu      = mu,
            user_bias  = user_bias,
            item_bias  = item_bias,
            all_singular_values = S_full,
            k       = k_eff,
        )
        res = evaluate_model(f"SVD  k={k:3d}", model_k.predict,
                              test_df, user_idx, item_idx)
        rmse_list.append(res.rmse)
        mae_list.append(res.mae)

        # Eckart-Young bound, normalised to per-entry RMSE scale
        ey_raw  = eckart_young_error(S_full, k_eff)
        ey_norm = ey_raw / np.sqrt(m_dim * n_dim)
        ey_list.append(ey_norm)

    # ── 8. Auto-select best k ─────────────────────────────────────────────────
    best_idx = int(np.argmin(rmse_list))
    BEST_K   = K_VALUES[best_idx]
    k_var    = select_k_by_variance(S_full, threshold=VAR_THRESH)

    logger.info("Auto-selected best k = %d  (test RMSE = %.4f)", BEST_K, rmse_list[best_idx])
    logger.info("Variance criterion (tau=%.0f%%): k = %d", VAR_THRESH * 100, k_var)

    k_final = min(BEST_K, len(S_full))
    best_model = SVDModel(
        U_k     = U_full[:, :k_final],
        sigma_k = S_full[:k_final],
        Vt_k    = Vt_full[:k_final, :],
        mu      = mu,
        user_bias  = user_bias,
        item_bias  = item_bias,
        all_singular_values = S_full,
        k       = k_final,
    )

    # ── 9. Fit and evaluate baselines ─────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Fitting baseline models...")
    logger.info("-" * 50)

    gm  = GlobalMeanBaseline().fit(R)
    um  = UserMeanBaseline().fit(R)
    im_ = ItemMeanBaseline().fit(R)
    uim = UserItemMeanBaseline().fit(R)

    eval_results = []
    for label, model in [
        ("Global Mean",          gm),
        ("User Mean",            um),
        ("Item Mean",            im_),
        ("User+Item Bias",       uim),
        (f"SVD  k={k_final}",    best_model),
    ]:
        eval_results.append(
            evaluate_model(label, model.predict, test_df, user_idx, item_idx)
        )

    if RUN_SGD:
        logger.info("Fitting SGD Matrix Factorisation (k=%d)...", BEST_K)
        sgd = SGDMatrixFactorization(
            k=BEST_K, lr=SGD_LR, reg=SGD_REG,
            epochs=SGD_EPOCHS, log_interval=SGD_LOG_INT,
        ).fit(R)
        eval_results.append(
            evaluate_model(f"SGD-MF k={BEST_K}", sgd.predict, test_df, user_idx, item_idx)
        )

    # ── 10. Results table ─────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  FINAL RESULTS TABLE")
    logger.info("=" * 55)
    table = results_table(eval_results)
    # print clean table to stdout even at INFO level
    print("\n" + table.to_string(index=False))

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_csv, index=False)
    logger.info("Results saved to %s", results_csv)

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Generating visualisation plots...")
    logger.info("-" * 50)

    plot_singular_value_decay(S_full, top_n=min(200, len(S_full)), plots_dir=plots_dir)
    plot_explained_variance(S_full, ks=K_VALUES, plots_dir=plots_dir)
    plot_rmse_vs_k(K_VALUES[:len(rmse_list)], rmse_list, ey_list, plots_dir=plots_dir)
    plot_mae_vs_k(K_VALUES[:len(mae_list)],   mae_list,            plots_dir=plots_dir)
    plot_rating_distribution(df,                                    plots_dir=plots_dir)
    plot_sparsity(R,                                                plots_dir=plots_dir)

    R_hat = best_model.reconstruct()
    plot_reconstruction_heatmap(R, R_hat, block=60, plots_dir=plots_dir)
    plot_baseline_comparison(table,                                 plots_dir=plots_dir)
    plot_latent_factors(best_model.U_k, best_model.Vt_k,           plots_dir=plots_dir)

    # ── 12. Summary ───────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", total_time)
    logger.info("  Best SVD rank  : k = %d",       BEST_K)
    logger.info("  Test RMSE      : %.4f",         rmse_list[best_idx])
    logger.info("  Test MAE       : %.4f",         mae_list[best_idx])
    logger.info("  k (90%% var)   : %d",           k_var)
    logger.info("  Plots saved to : %s",           plots_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
