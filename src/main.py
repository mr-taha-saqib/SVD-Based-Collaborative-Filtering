"""
main.py
=======
End-to-end training pipeline for the SVD recommender (corrected methodology).

    python src/main.py                      # train, evaluate, register a candidate
    python src/main.py --approve            # ... and approve it for serving
    python src/main.py --no-plots --no-mlflow

Pipeline stages
---------------
 1. Load configuration and set up logging
 2. Download (checksum-verified) and validate MovieLens ratings
 3. Per-user stratified TRAIN / VALIDATION / TEST split + leakage verification
 4. Build the train matrix (train vocabulary only), bias-centre, impute
 5. Compute the SVD once and sweep rank k on VALIDATION
 6. Select k by minimum VALIDATION RMSE  (the historical version used TEST here)
 7. Fit baselines and SGD-MF on TRAIN; compare everything on VALIDATION
 8. Ranking metrics (Precision/Recall/NDCG@K, coverage) on VALIDATION
 9. FINAL evaluation of the already-chosen configuration on TEST (exactly once)
10. Save results, plots, data-quality and split manifests to outputs/corrected/
11. Build the serving artifact and register it as a new model version
12. Record everything in MLflow; approve for serving only with --approve

Historical results from the previous method are kept untouched in outputs/ and
labelled in outputs/HISTORICAL_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (  # noqa: E402
    dataset_fingerprint,
    load_movies,
    load_ratings_from_config,
    train_val_test_split,
    validate_ratings,
)
from evaluation import (  # noqa: E402
    GlobalMeanBaseline,
    ItemMeanBaseline,
    SGDMatrixFactorization,
    UserItemMeanBaseline,
    UserMeanBaseline,
)
from ranking import bias_scores, popularity_scores  # noqa: E402
from recommender import build_artifact  # noqa: E402
from registry import ModelRegistry  # noqa: E402
from settings import get, load_config, resolve_path  # noqa: E402
from svd_model import select_k_by_variance  # noqa: E402
from tracking import Tracker, git_commit  # noqa: E402
from training import decompose, rating_and_ranking_metrics, select_rank  # noqa: E402

logger = logging.getLogger("main")


def environment_info() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "mlflow"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = "not installed"
    return info


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    cfg: dict[str, Any],
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    make_plots: bool = True,
    register: bool = True,
    approve: bool = False,
    approval_reason: str | None = None,
) -> dict[str, Any]:
    """Run the corrected pipeline. Returns a summary dict (also written to disk).

    ``ratings``/``movies`` may be passed directly (tests, synthetic fixtures);
    otherwise the configured MovieLens files are loaded (and downloaded).
    """
    t_total = time.perf_counter()
    out_dir = Path(out_dir) if out_dir else resolve_path(get(cfg, "output.corrected_dir", "outputs/corrected"))
    out_dir.mkdir(parents=True, exist_ok=True)

    split_cfg, model_cfg = cfg.get("split", {}), cfg.get("model", {})
    rank_cfg, sgd_cfg = cfg.get("ranking", {}), cfg.get("sgd_mf", {})
    k_values = [int(k) for k in model_cfg.get("k_values", [5, 10, 20, 50, 100])]
    seed = int(split_cfg.get("random_state", 42))
    rk = int(rank_cfg.get("k", 10))

    # ── 2. data ───────────────────────────────────────────────────────────────
    if ratings is None:
        ratings, dq = load_ratings_from_config(cfg)
    else:
        ratings, dq = validate_ratings(ratings, rating_scale=tuple(get(cfg, "dataset.rating_scale", [0.5, 5.0])))
    if movies is None:
        movies = load_movies(resolve_path(get(cfg, "dataset.movies_path", "data/ml-latest-small/movies.csv")))
    write_json(dq.to_dict(), out_dir / "data_quality.json")
    data_sha = dataset_fingerprint(ratings)

    # ── 3. split ──────────────────────────────────────────────────────────────
    split = train_val_test_split(ratings, validation_frac=float(split_cfg.get("validation_fraction", 0.10)),
                                 test_frac=float(split_cfg.get("test_fraction", 0.20)), random_state=seed)
    write_json(split.manifest, out_dir / "split_manifest.json")
    train_df, val_df, test_df = split.train, split.validation, split.test

    # ── 4-6. decomposition + validation rank selection ────────────────────────
    dec = decompose(train_df, max(k_values), imputation=get(cfg, "preprocessing.imputation_strategy", "zero"),
                    solver=str(model_cfg.get("svd_solver", "auto")), random_state=seed)
    best_k, sweep = select_rank(dec, k_values, val_df)          # VALIDATION only
    sweep.to_csv(out_dir / "k_sweep_validation.csv", index=False)
    k_var = select_k_by_variance(dec.S, threshold=float(model_cfg.get("variance_threshold", 0.9)))
    svd = dec.model(best_k)

    # ── 7. baselines + SGD-MF on TRAIN ────────────────────────────────────────
    R = dec.R
    models: list[tuple[str, Any, Any]] = [
        ("Global Mean", GlobalMeanBaseline().fit(R), None),
        ("User Mean", UserMeanBaseline().fit(R), None),
        ("Item Mean", ItemMeanBaseline().fit(R), None),
        ("User+Item Bias", UserItemMeanBaseline().fit(R), bias_scores(dec.mu, dec.user_bias, dec.item_bias)),
        (f"SVD k={best_k}", svd, svd.score_users),
    ]
    if bool(sgd_cfg.get("enabled", True)):
        sgd = SGDMatrixFactorization(k=best_k, lr=float(sgd_cfg.get("learning_rate", 0.005)),
                                     reg=float(sgd_cfg.get("regularization", 0.02)),
                                     epochs=int(sgd_cfg.get("epochs", 20)),
                                     log_interval=int(sgd_cfg.get("log_interval", 5)),
                                     seed=int(sgd_cfg.get("seed", 42))).fit(R)
        models.append((f"SGD-MF k={best_k}", sgd, None))
    popularity = ("Most Popular", None, popularity_scores(dec.item_counts))

    # ── 7-8. VALIDATION comparison ────────────────────────────────────────────
    val_rows = [rating_and_ranking_metrics(n, m, "validation", val_df, train_df, dec, sf, rank_cfg)
                for n, m, sf in models]
    val_rows.append(_ranking_only(popularity, "validation", val_df, train_df, dec, rank_cfg))
    val_table = pd.DataFrame(val_rows)
    val_table.to_csv(out_dir / "results_validation.csv", index=False)

    # ── 9. FINAL TEST evaluation (configuration already fixed) ────────────────
    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION: k=%d was fixed on validation before this point", best_k)
    logger.info("=" * 60)
    known_for_test = pd.concat([train_df, val_df], ignore_index=True)
    test_rows = [rating_and_ranking_metrics(n, m, "test", test_df, known_for_test, dec, sf, rank_cfg)
                 for n, m, sf in models]
    test_rows.append(_ranking_only(popularity, "test", test_df, known_for_test, dec, rank_cfg))
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(out_dir / "results_test.csv", index=False)

    svd_val = val_table.loc[val_table["model"] == f"SVD k={best_k}"].iloc[0].to_dict()
    svd_test = test_table.loc[test_table["model"] == f"SVD k={best_k}"].iloc[0].to_dict()
    summary: dict[str, Any] = {
        "methodology": "train/validation/test; rank selected on validation RMSE; test evaluated once",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"name": get(cfg, "dataset.name", "custom"), "sha256": data_sha,
                    "n_ratings": int(len(ratings)), "n_users": int(ratings["userId"].nunique()),
                    "n_items": int(ratings["movieId"].nunique())},
        "split": split.manifest,
        "selection": {"metric": "validation RMSE", "k_values": k_values, "selected_k": best_k,
                      "k_by_90pct_variance": int(k_var), "svd_solver": dec.solver},
        "validation": {"svd": svd_val, "all_models": val_rows},
        "test": {"svd": svd_test, "all_models": test_rows},
        "environment": environment_info(),
        "git_commit": git_commit(),
    }
    write_json(summary, out_dir / "final_metrics.json")
    _log_tables(val_table, test_table, rk)

    if make_plots:
        _make_plots(ratings, dec, sweep, best_k, test_table, out_dir, cfg)

    # ── 11-12. artifact, registry, MLflow ─────────────────────────────────────
    tracker = Tracker(cfg)
    version_id = None
    with tracker.run(run_name=f"svd-k{best_k}-train", tags={"stage": "initial_training",
                                                           "dataset_sha256": data_sha}):
        tracker.log_params({"algorithm": "truncated SVD (bias-centred, zero-imputed)",
                            "selected_k": best_k, "k_values": k_values,
                            "selection_metric": "validation_rmse", "seed": seed,
                            "validation_fraction": split_cfg.get("validation_fraction"),
                            "test_fraction": split_cfg.get("test_fraction"),
                            "imputation": get(cfg, "preprocessing.imputation_strategy", "zero"),
                            "svd_solver": dec.solver, "dataset_sha256": data_sha,
                            "split_sha256": split.manifest["split_sha256"],
                            "sgd_lr": sgd_cfg.get("learning_rate"), "sgd_reg": sgd_cfg.get("regularization"),
                            "sgd_epochs": sgd_cfg.get("epochs"), "ranking_k": rk,
                            "min_item_ratings": rank_cfg.get("min_item_ratings")})
        for _, r in sweep.iterrows():
            tracker.log_metrics({"sweep_val_rmse": r["val_rmse"], "sweep_val_mae": r["val_mae"]}, step=int(r["k"]))
        tracker.log_metrics(_prefixed(svd_val, "val_"))
        tracker.log_metrics(_prefixed(svd_test, "test_"))
        for f in ("final_metrics.json", "k_sweep_validation.csv", "results_validation.csv",
                  "results_test.csv", "split_manifest.json", "data_quality.json"):
            tracker.log_artifact(out_dir / f, artifact_path="reports")

        if register:
            seen = pd.concat([train_df, val_df, test_df], ignore_index=True)
            with tempfile.TemporaryDirectory() as tmp:
                meta = {
                    "model_name": get(cfg, "registry.model_name", "svd-movie-recommender"),
                    "algorithm": "truncated SVD with additive bias decomposition",
                    "k": best_k, "created_at": summary["created_at"],
                    "training_data": {"dataset_sha256": data_sha,
                                      "split_sha256": split.manifest["split_sha256"],
                                      "trained_on": "train partition only",
                                      "n_train_ratings": int(len(train_df))},
                    "metrics": {"validation": svd_val, "test": svd_test},
                    "config": {k: cfg.get(k) for k in ("split", "model", "ranking", "preprocessing")},
                    "environment": summary["environment"], "git_commit": summary["git_commit"],
                    "mlflow_run_id": tracker.run_id,
                }
                content_sha = build_artifact(tmp, svd, dec.user_ids, dec.item_ids, train_df, seen, movies, meta,
                                             min_item_ratings=int(rank_cfg.get("min_item_ratings", 5)),
                                             reference_top_k=rk)
                version_id = f"svd-k{best_k}-{data_sha[:8]}-{content_sha[:8]}"
                meta_written = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
                meta_written["version_id"] = version_id
                reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "models")))
                vdir = reg.register(tmp, version_id, meta_written)
            tracker.log_artifacts(vdir, artifact_path="model")
            tracker.set_tags({"registry_version": version_id})
            mlflow_version = tracker.register_model(version_id)
            if approve:
                reason = approval_reason or (
                    "Initial approval by operator after review of validation-based selection "
                    "and single final test evaluation")
                reg.approve(version_id, {"reason": reason, "type": "manual_initial_approval",
                                         "validation_rmse": svd_val["rmse"], "test_rmse": svd_test["rmse"],
                                         "mlflow_run_id": tracker.run_id})
                tracker.set_alias("champion", mlflow_version)
                tracker.set_tags({"approved": "true"})
            summary["registry"] = {"version_id": version_id, "approved": bool(approve),
                                   "mlflow_model_version": mlflow_version}
    summary["mlflow_run_id"] = tracker.run_id
    summary["seconds"] = round(time.perf_counter() - t_total, 1)
    write_json(summary, out_dir / "final_metrics.json")
    logger.info("Pipeline complete in %.1fs | k=%d | val RMSE %.4f | test RMSE %.4f | version %s",
                summary["seconds"], best_k, svd_val["rmse"], svd_test["rmse"], version_id)
    return summary


def _ranking_only(entry, partition, heldout, known, dec, rank_cfg) -> dict:
    from ranking import evaluate_ranking
    from training import candidate_mask
    name, _, score_fn = entry
    rr = evaluate_ranking(name, partition, score_fn, heldout, known, dec.user_idx, dec.item_idx,
                          candidate_mask(dec.item_counts, rank_cfg.get("min_item_ratings", 5)),
                          k=int(rank_cfg.get("k", 10)),
                          relevance_threshold=float(rank_cfg.get("relevance_threshold", 4.0)))
    return {"model": name, "partition": partition, "rmse": None, "mae": None, "n": None,
            "n_fallback": None, f"precision@{rr.k}": rr.precision, f"recall@{rr.k}": rr.recall,
            f"ndcg@{rr.k}": rr.ndcg, f"hit_rate@{rr.k}": rr.hit_rate,
            "catalog_coverage": rr.catalog_coverage, "ranking_users": rr.n_users_evaluated}


def _prefixed(d: dict, prefix: str) -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and v is not None and not (
                isinstance(v, float) and np.isnan(v)):
            out[prefix + k.replace("@", "_at_")] = float(v)
    return out


def _log_tables(val_table: pd.DataFrame, test_table: pd.DataFrame, rk: int) -> None:
    cols = ["model", "rmse", "mae", f"precision@{rk}", f"recall@{rk}", f"ndcg@{rk}", "catalog_coverage"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print("\nVALIDATION (model selection)\n" + val_table[cols].to_string(index=False))
        print("\nTEST (final, evaluated once)\n" + test_table[cols].to_string(index=False))


def _make_plots(ratings, dec, sweep, best_k, test_table, out_dir, cfg) -> None:
    from visualizations import (
        plot_baseline_comparison,
        plot_explained_variance,
        plot_latent_factors,
        plot_mae_vs_k,
        plot_rating_distribution,
        plot_rmse_vs_k,
        plot_singular_value_decay,
        plot_sparsity,
    )
    plots_dir = resolve_path(get(cfg, "output.plots_dir", str(out_dir / "plots")))
    ks = sweep["k"].tolist()
    ey = sweep["ey_bound_normalised"].tolist()
    plot_singular_value_decay(dec.S, top_n=min(200, len(dec.S)), plots_dir=plots_dir)
    plot_explained_variance(dec.S, ks=ks, plots_dir=plots_dir)
    plot_rmse_vs_k(ks, sweep["val_rmse"].tolist(), None if np.isnan(ey).any() else ey,
                   plots_dir=plots_dir, partition="Validation")
    plot_mae_vs_k(ks, sweep["val_mae"].tolist(), plots_dir=plots_dir, partition="Validation")
    plot_rating_distribution(ratings, plots_dir=plots_dir)
    plot_sparsity(dec.R, plots_dir=plots_dir)
    rated = test_table.dropna(subset=["rmse"]).rename(columns={"model": "Model", "rmse": "RMSE", "mae": "MAE"})
    plot_baseline_comparison(rated[["Model", "RMSE", "MAE"]], plots_dir=plots_dir)
    m = dec.model(best_k)
    plot_latent_factors(m.U_k, m.Vt_k, plots_dir=plots_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to config.yaml (default: project root or $SVD_CONFIG)")
    p.add_argument("--ratings", help="override ratings CSV (userId,movieId,rating[,timestamp])")
    p.add_argument("--movies", help="override movies CSV (movieId,title,genres)")
    p.add_argument("--out-dir", help="override output directory")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-register", action="store_true", help="do not add the model to the registry")
    p.add_argument("--approve", action="store_true",
                   help="approve the new version for serving (explicit operator decision)")
    p.add_argument("--approval-reason", help="free-text reason recorded with the approval")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from logging_utils import setup_logging
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.no_mlflow:
        cfg.setdefault("mlflow", {})["enabled"] = False
    out_dir = Path(args.out_dir) if args.out_dir else resolve_path(get(cfg, "output.corrected_dir"))
    setup_logging(get(cfg, "output.log_level", "INFO"), out_dir / "run.log")
    logger.info("SVD-Based Collaborative Filtering - corrected train/validation/test pipeline")
    ratings = movies = None
    if args.ratings:
        ratings = pd.read_csv(args.ratings)
    if args.movies:
        movies = load_movies(Path(args.movies))
    run_pipeline(cfg, ratings=ratings, movies=movies, out_dir=out_dir, make_plots=not args.no_plots,
                 register=not args.no_register, approve=args.approve, approval_reason=args.approval_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
