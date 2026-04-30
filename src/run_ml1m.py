"""
run_ml1m.py
===========
Cross-dataset validation pipeline: runs the identical SVD methodology on
MovieLens 1M (6,040 users x 3,706 items, ~1M ratings) and produces a
cross-dataset comparison figure alongside ML-small results.

Key differences vs main.py:
  - Uses sklearn randomised SVD (n_components=220) for the larger matrix.
    This is O(m*n*k) vs O(m*n*min(m,n)) for full SVD — much faster.
  - SGD-MF disabled (20 epochs x 1M ratings would take ~30 min).
  - Rating clip set to [1.0, 5.0] matching ML-1M integer scale.
  - Saves results to outputs/results_ml1m.csv.
  - Loads ML-small results from outputs/results_table.csv for comparison.
  - Generates outputs/plots/10_crossdataset_comparison.png.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_ratings_ml1m,
    train_test_split,
    build_user_item_matrix,
    center_matrix,
    impute_matrix,
)
from svd_model import SVDModel, select_k_by_variance, eckart_young_error
from evaluation import (
    evaluate_model,
    results_table,
    GlobalMeanBaseline,
    UserMeanBaseline,
    ItemMeanBaseline,
    UserItemMeanBaseline,
)
from visualizations import plot_crossdataset_comparison

# ── config ────────────────────────────────────────────────────────────────────
K_VALUES     = [5, 10, 20, 50, 100, 200]
N_COMPONENTS = 220          # randomised SVD components (buffer above k_max=200)
TEST_FRAC    = 0.20
RANDOM_STATE = 42
VAR_THRESH   = 0.90
RATING_CLIP  = (1.0, 5.0)   # ML-1M uses integer ratings 1-5

RESULTS_DIR  = Path(__file__).parent.parent / "outputs"
PLOTS_DIR    = RESULTS_DIR / "plots"
RESULTS_CSV  = RESULTS_DIR / "results_ml1m.csv"
SMALL_CSV    = RESULTS_DIR / "results_table.csv"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("run_ml1m")

    logger.info("=" * 60)
    logger.info("  Cross-Dataset Validation: MovieLens 1M Pipeline")
    logger.info("=" * 60)
    t_total = time.perf_counter()

    # ── 1. Load ML-1M ─────────────────────────────────────────────────────────
    df = load_ratings_ml1m()

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_frac=TEST_FRAC,
                                         random_state=RANDOM_STATE)

    # ── 3. Build matrix ───────────────────────────────────────────────────────
    R, user_idx, item_idx = build_user_item_matrix(train_df, full_df=df)

    # ── 4. Centre + impute ────────────────────────────────────────────────────
    R_c, mu, user_bias, item_bias = center_matrix(R)
    R_filled = impute_matrix(R_c, strategy="zero")

    # ── 5. Randomised SVD (top N_COMPONENTS) ──────────────────────────────────
    logger.info("Randomised SVD: matrix %s, n_components=%d ...",
                R_filled.shape, N_COMPONENTS)
    t_svd = time.perf_counter()
    U_full, S_full, Vt_full = randomized_svd(
        R_filled, n_components=N_COMPONENTS,
        n_iter=4, random_state=RANDOM_STATE,
    )
    logger.info("Randomised SVD done in %.1fs | top sigma=%.2f | sigma_%d=%.4f",
                time.perf_counter() - t_svd, S_full[0], N_COMPONENTS, S_full[-1])

    # ── 6. k-sweep ────────────────────────────────────────────────────────────
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
            U_k=U_full[:, :k_eff], sigma_k=S_full[:k_eff],
            Vt_k=Vt_full[:k_eff, :], mu=mu,
            user_bias=user_bias, item_bias=item_bias,
            all_singular_values=S_full, k=k_eff,
        )
        res = evaluate_model(f"SVD  k={k:3d}",
                             lambda u, i, m=model_k: m.predict(u, i, clip=RATING_CLIP),
                             test_df, user_idx, item_idx)
        rmse_list.append(res.rmse)
        mae_list.append(res.mae)

        ey_raw  = eckart_young_error(S_full, k_eff)
        ey_norm = ey_raw / np.sqrt(m_dim * n_dim)
        ey_list.append(ey_norm)

    best_idx = int(np.argmin(rmse_list))
    BEST_K   = K_VALUES[best_idx]
    k_var    = select_k_by_variance(S_full, threshold=VAR_THRESH)
    logger.info("Best k = %d  (RMSE=%.4f)  |  k(90%% EV) = %d",
                BEST_K, rmse_list[best_idx], k_var)

    # ── 7. Baselines ──────────────────────────────────────────────────────────
    logger.info("Fitting baselines...")
    k_eff   = min(BEST_K, len(S_full))
    best_model = SVDModel(
        U_k=U_full[:, :k_eff], sigma_k=S_full[:k_eff],
        Vt_k=Vt_full[:k_eff, :], mu=mu,
        user_bias=user_bias, item_bias=item_bias,
        all_singular_values=S_full, k=k_eff,
    )

    eval_results = []
    baselines = [
        ("Global Mean",    GlobalMeanBaseline().fit(R)),
        ("User Mean",      UserMeanBaseline().fit(R)),
        ("Item Mean",      ItemMeanBaseline().fit(R)),
        ("User+Item Bias", UserItemMeanBaseline().fit(R)),
    ]
    for label, mdl in baselines:
        eval_results.append(
            evaluate_model(label, mdl.predict, test_df, user_idx, item_idx)
        )
    eval_results.append(
        evaluate_model(
            f"SVD  k={BEST_K}",
            lambda u, i: best_model.predict(u, i, clip=RATING_CLIP),
            test_df, user_idx, item_idx,
        )
    )

    # ── 8. Save results ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  ML-1M RESULTS TABLE")
    logger.info("=" * 55)
    table = results_table(eval_results)
    print("\n" + table.to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    logger.info("Results saved -> %s", RESULTS_CSV)

    # ── 9. Cross-dataset comparison plot ──────────────────────────────────────
    if SMALL_CSV.exists():
        small_table = pd.read_csv(SMALL_CSV)
        logger.info("Generating cross-dataset comparison plot...")
        svd_rows = small_table[small_table["Model"].str.startswith("SVD")]
        if len(svd_rows) > 0:
            small_best_rmse = float(svd_rows["RMSE"].iloc[0])
            small_best_k    = int(svd_rows["Model"].iloc[0].split("k=")[-1].strip())
        else:
            small_best_rmse = float(small_table["RMSE"].min())
            small_best_k    = 10
        plot_crossdataset_comparison(
            small_ks        = [small_best_k],
            small_rmse      = [small_best_rmse],
            m1_ks           = K_VALUES,
            m1_rmse         = rmse_list,
            small_baselines = small_table,
            m1_baselines    = table,
            plots_dir       = PLOTS_DIR,
        )
    else:
        logger.warning("ML-small results CSV not found; skipping comparison plot.")

    total = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("ML-1M pipeline complete in %.1fs", total)
    logger.info("  Users: %d | Items: %d | Ratings: %d",
                df["userId"].nunique(), df["movieId"].nunique(), len(df))
    logger.info("  Best SVD rank : k = %d", BEST_K)
    logger.info("  Test RMSE     : %.4f", rmse_list[best_idx])
    logger.info("  Test MAE      : %.4f", mae_list[best_idx])
    logger.info("  k (90%% EV)  : %d", k_var)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
