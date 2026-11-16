"""Train and approve a model on SYNTHETIC data into a given registry directory.

Used by CI and the Docker smoke test so they need neither the MovieLens download
nor the real model. The resulting model is a demonstration artifact only.

    python scripts/build_demo_model.py --registry ci-models
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_data import make_movies, make_ratings  # noqa: E402

from main import run_pipeline  # noqa: E402
from settings import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="registry directory to create")
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--items", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(apply_env=False)
    cfg["registry"]["dir"] = str(Path(args.registry).resolve())
    cfg["mlflow"]["enabled"] = False
    cfg["model"]["k_values"] = [2, 3, 5, 8]
    cfg["sgd_mf"]["enabled"] = False
    cfg["ranking"]["min_item_ratings"] = 2
    ratings = make_ratings(args.users, args.items, 0.25, seed=1)
    with tempfile.TemporaryDirectory() as out:
        s = run_pipeline(cfg, ratings=ratings, movies=make_movies(ratings), out_dir=Path(out),
                         make_plots=False, register=True, approve=True,
                         approval_reason="synthetic demo model for CI / smoke tests")
    print(s["registry"]["version_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
