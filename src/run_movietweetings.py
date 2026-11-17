"""
run_movietweetings.py
=====================
Separate pipeline: the corrected methodology (train / validation / test, rank selected on
validation RMSE only, test evaluated once, most-popular and bias baselines, ranking metrics)
applied to MovieTweetings instead of MovieLens.

    python src/run_movietweetings.py                    # download (checksum-pinned), train, evaluate
    python src/run_movietweetings.py --approve --approval-reason "..."   # approve in ITS OWN registry
    python src/run_movietweetings.py --retrain-feedback fb.csv [--dry-run]  # gated retraining
    python src/run_movietweetings.py --rollback                            # re-approve previous version

Everything is written to separate locations, so the MovieLens results, registry and MLflow
experiment are untouched:
  outputs/external/movietweetings/        metrics, split manifest, data preparation report
  models_external/movietweetings/         separate model registry
  MLflow experiment "svd-movietweetings"  (same tracking URI as the main pipeline)

Differences from the MovieLens run (all fixed before fitting, see src/movietweetings.py):
ratings / 2, ratings of 0 dropped, 20-core filter, SGD-MF disabled (pure-Python SGD over
~0.6M ratings is slow and was likewise disabled for MovieLens 1M), extended k grid.

LICENCE: MovieTweetings is MIT-licensed by its author, but it is derived from public tweets
and refers to IMDb titles. Commercial use still requires the client's legal review
(docs/DATASET_LICENSING.md). Approving a model here is a technical step, not a licence decision.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import setup_logging  # noqa: E402
from main import run_pipeline  # noqa: E402
from movietweetings import COMMIT, load_movietweetings  # noqa: E402
from settings import load_config, resolve_path  # noqa: E402

K_VALUES = [5, 10, 20, 50, 100, 150, 200]


def build_config(base: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["dataset"] = {"name": f"movietweetings@{COMMIT[:7]} (20-core, rating/2)", "rating_scale": [0.5, 5.0]}
    cfg.setdefault("model", {})["k_values"] = K_VALUES
    cfg["model"]["svd_solver"] = "randomized"
    cfg.setdefault("sgd_mf", {})["enabled"] = False
    cfg["output"] = dict(cfg.get("output", {}))
    cfg["output"]["corrected_dir"] = "outputs/external/movietweetings"
    cfg["output"]["plots_dir"] = "outputs/external/movietweetings/plots"
    cfg["registry"] = {"dir": "models_external/movietweetings", "model_name": "svd-movietweetings"}
    cfg.setdefault("mlflow", {})["experiment"] = "svd-movietweetings"
    cfg.setdefault("retraining", {})["report_dir"] = "outputs/external/movietweetings/retraining_reports"
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default="data/movietweetings")
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--approval-reason", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--retrain-feedback", help="CSV of new ratings on the 0.5-5 scale; runs gated retraining")
    ap.add_argument("--dry-run", action="store_true", help="with --retrain-feedback: evaluate, change nothing")
    ap.add_argument("--rollback", action="store_true", help="re-approve the previous approved version")
    ap.add_argument("--registry-dir", help="override the separate registry location (e.g. a copy for tests)")
    args = ap.parse_args(argv)

    cfg = build_config(load_config())
    if args.registry_dir:
        cfg["registry"]["dir"] = args.registry_dir
    out_dir = resolve_path(cfg["output"]["corrected_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", out_dir / "run.log")
    log = logging.getLogger("run_movietweetings")

    if args.rollback:
        from registry import ModelRegistry, RegistryError
        try:
            v = ModelRegistry(resolve_path(cfg["registry"]["dir"])).rollback("manual rollback (MovieTweetings)")
        except RegistryError as exc:
            log.error("Rollback not possible: %s", exc)
            return 1
        from tracking import Tracker
        Tracker(cfg).mirror_alias("champion", v)    # keep the MLflow alias in step with the registry
        print(json.dumps({"rolled_back_to": v}))
        return 0

    ratings, movies, prep = load_movietweetings(resolve_path(args.data_dir))
    if args.retrain_feedback:
        import pandas as pd

        from retrain import RetrainingError, run_retraining
        try:
            rep = run_retraining(cfg, pd.read_csv(args.retrain_feedback), dry_run=args.dry_run,
                                 base_ratings=ratings, movies=movies)
        except RetrainingError as exc:
            log.error("Retraining failed: %s", exc)
            return 1
        print(json.dumps({k: rep.get(k) for k in ("decision", "promoted", "approved_version",
                                                   "candidate_version", "approved_after")}, indent=2))
        return 0
    (out_dir / "data_preparation.json").write_text(json.dumps(prep.to_dict(), indent=2), encoding="utf-8")
    summary = run_pipeline(cfg, ratings=ratings, movies=movies, out_dir=out_dir, make_plots=not args.no_plots,
                           register=True, approve=args.approve, approval_reason=args.approval_reason)
    summary["data_preparation"] = prep.to_dict()
    summary["licence_note"] = ("MovieTweetings (MIT, Simon Dooms). Derived from public tweets; IMDb ids. "
                               "Commercial use requires the client's legal review; see docs/DATASET_LICENSING.md.")
    (out_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("MovieTweetings pipeline complete: k=%s", summary["selection"]["selected_k"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
