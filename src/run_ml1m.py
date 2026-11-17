"""
run_ml1m.py
===========
Cross-dataset validation on MovieLens 1M (6,040 users x 3,706 items, ~1M ratings)
using the CORRECTED methodology: train/validation/test split, rank selected on
validation, test evaluated once.

    python src/run_ml1m.py

Differences from the ML-small pipeline (unchanged from the original study):
  * randomised SVD (sklearn) with n_components = max(k) + 20, O(m*n*k) instead of
    a full dense SVD;
  * SGD-MF disabled (pure-Python SGD over 1M ratings is too slow to be useful here);
  * rating clip [1, 5] (ML-1M uses integer ratings).

Writes to outputs/corrected/ml1m/. The historical outputs/results_ml1m.csv (which
selected k on the test set) is left untouched; see outputs/HISTORICAL_RESULTS.md.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import dataset_fingerprint, load_ratings_ml1m, train_val_test_split, validate_ratings  # noqa: E402
from evaluation import (  # noqa: E402
    GlobalMeanBaseline,
    ItemMeanBaseline,
    UserItemMeanBaseline,
    UserMeanBaseline,
    evaluate_pairs,
)
from logging_utils import setup_logging  # noqa: E402
from settings import get, load_config, resolve_path  # noqa: E402
from training import decompose, select_rank  # noqa: E402

K_VALUES = [5, 10, 20, 50, 100, 200]
RATING_CLIP = (1.0, 5.0)


class _Clipped:
    """Wrap a model so predictions use the ML-1M rating range."""

    def __init__(self, model):
        self.model = model

    def predict_batch(self, u, i):
        return self.model.predict_batch(u, i, clip=RATING_CLIP)

    def fallback_batch(self, u, i):
        return self.model.fallback_batch(u, i, clip=RATING_CLIP)


def main() -> int:
    cfg = load_config()
    out_dir = resolve_path(get(cfg, "output.corrected_dir", "outputs/corrected")) / "ml1m"
    setup_logging("INFO", out_dir / "run.log")
    logger = logging.getLogger("run_ml1m")
    t0 = time.perf_counter()
    split_cfg = cfg.get("split", {})

    df, dq = validate_ratings(load_ratings_ml1m(), rating_scale=(1.0, 5.0), rating_step=1.0)
    sp = train_val_test_split(df, float(split_cfg.get("validation_fraction", 0.1)),
                              float(split_cfg.get("test_fraction", 0.2)), int(split_cfg.get("random_state", 42)))
    dec = decompose(sp.train, max(K_VALUES), solver="randomized", random_state=int(split_cfg.get("random_state", 42)))

    # model selection on VALIDATION only (clipping to the ML-1M scale)
    best_k, sweep = select_rank(_ClippedDecomposition(dec), K_VALUES, sp.validation)
    sweep.to_csv(out_dir / "k_sweep_validation.csv", index=False)

    models = [("Global Mean", GlobalMeanBaseline().fit(dec.R)), ("User Mean", UserMeanBaseline().fit(dec.R)),
              ("Item Mean", ItemMeanBaseline().fit(dec.R)), ("User+Item Bias", UserItemMeanBaseline().fit(dec.R)),
              (f"SVD k={best_k}", _Clipped(dec.model(best_k)))]
    rows = []
    for part, frame in (("validation", sp.validation), ("test", sp.test)):
        for name, m in models:
            r = evaluate_pairs(name, m, frame, dec.user_idx, dec.item_idx)
            rows.append({"model": name, "partition": part, "rmse": r.rmse, "mae": r.mae, "n": r.n_samples,
                         "n_fallback": r.n_fallback})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "results.csv", index=False)
    summary = {"dataset": "MovieLens 1M", "dataset_sha256": dataset_fingerprint(df),
               "data_quality": dq.to_dict(), "split": sp.manifest, "selected_k": best_k,
               "selection_metric": "validation RMSE", "svd_solver": dec.solver,
               "results": rows, "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(table.to_string(index=False))
    logger.info("ML-1M corrected pipeline complete: k=%d (validation-selected) in %.1fs",
                best_k, summary["seconds"])
    return 0


class _ClippedDecomposition:
    """Adapter so select_rank evaluates models with the ML-1M clip range."""

    def __init__(self, dec):
        self._dec = dec
        self.R, self.S, self.solver = dec.R, dec.S, dec.solver
        self.user_idx, self.item_idx = dec.user_idx, dec.item_idx

    def model(self, k):
        inner = self._dec.model(k)
        wrapped = _Clipped(inner)
        wrapped.k = inner.k
        return wrapped


if __name__ == "__main__":
    raise SystemExit(main())
