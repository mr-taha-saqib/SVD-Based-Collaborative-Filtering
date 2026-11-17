"""
Generate SYNTHETIC feedback ratings to exercise the retraining workflow.

THIS IS DEMONSTRATION DATA, NOT REAL USER BEHAVIOUR. Real production retraining
requires genuinely observed ratings.

Modes
-----
consistent : ratings drawn around the approved model's own predictions (+ noise)
             for movies users have not rated: a plausible "more of the same" signal.
noise      : uniformly random ratings: no signal; a candidate trained on it
             should not be promoted.
new_users  : ratings from NEW user ids (not in the model). Each newcomer copies the
             taste profile of a random existing user (that user's predicted ratings
             + noise). The approved model can only use biases for them, so a
             retrained candidate is expected to improve on validation.
invalid    : 'consistent' data plus malformed rows (bad ids, out-of-range,
             missing values, duplicates) to exercise validation.

    python scripts/generate_synthetic_feedback.py --mode consistent --n 3000 --out data/feedback/synthetic_consistent.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import Recommender  # noqa: E402
from registry import ModelRegistry  # noqa: E402
from settings import get, load_config, resolve_path  # noqa: E402


def generate(rec: Recommender, n: int, mode: str, seed: int = 7, noise_sd: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = rec.user_ids
    candidates = np.flatnonzero(rec.candidate_mask)
    rows, seen = [], set()
    attempts = 0
    if mode == "new_users":
        next_id = int(users.max()) + 1
        while len(rows) < n:
            persona = int(rng.choice(users))
            n_items = int(rng.integers(30, 80))
            cols = rng.choice(candidates, size=min(n_items, len(candidates)), replace=False)
            for c in cols:
                pred, _ = rec.predict_rating(persona, int(rec.item_ids[c]))
                r = float(np.clip(np.round((pred + rng.normal(0, noise_sd)) * 2) / 2, 0.5, 5.0))
                rows.append({"userId": next_id, "movieId": int(rec.item_ids[c]), "rating": r,
                             "timestamp": 1700000000 + len(rows)})
            next_id += 1
        return pd.DataFrame(rows[:n])
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        u = int(rng.choice(users))
        c = int(rng.choice(candidates))
        m = int(rec.item_ids[c])
        if (u, m) in seen or m in set(rec.rated_items(u)):
            continue
        seen.add((u, m))
        if mode == "noise":
            r = float(rng.choice(np.arange(0.5, 5.01, 0.5)))
        else:
            pred, _ = rec.predict_rating(u, m)
            r = float(np.clip(np.round((pred + rng.normal(0, noise_sd)) * 2) / 2, 0.5, 5.0))
        rows.append({"userId": u, "movieId": m, "rating": r, "timestamp": 1700000000 + len(rows)})
    df = pd.DataFrame(rows)
    if mode == "invalid":
        bad = pd.DataFrame([
            {"userId": -1, "movieId": 1, "rating": 4.0, "timestamp": 0},
            {"userId": 1, "movieId": 1, "rating": 9.0, "timestamp": 0},
            {"userId": None, "movieId": 2, "rating": 3.0, "timestamp": 0},
            {"userId": 1.5, "movieId": 3, "rating": 3.0, "timestamp": 0},
        ])
        df = pd.concat([df, bad, df.head(5)], ignore_index=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["consistent", "new_users", "noise", "invalid"], default="consistent")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "models")))
    rec = Recommender.load(reg.approved_dir())
    df = generate(rec, args.n, args.mode, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC feedback rows ({args.mode}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
