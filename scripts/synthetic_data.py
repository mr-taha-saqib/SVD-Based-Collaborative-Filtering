"""SYNTHETIC MovieLens-shaped data for tests, CI and Docker smoke tests.

Low-rank structure plus noise on the 0.5..5.0 half-star scale, so the model has a
real (synthetic) signal to learn. Never presented as real user data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ratings(n_users: int = 80, n_items: int = 120, density: float = 0.3, rank: int = 3,
                 seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, rank))
    V = rng.normal(0, 1, (n_items, rank))
    item_pop = rng.gamma(1.0, 1.0, n_items)
    p = item_pop / item_pop.sum()
    rows = []
    for u in range(n_users):
        k = max(8, int(density * n_items))
        for i in rng.choice(n_items, size=k, replace=False, p=p):
            r = 3.4 + 0.6 * U[u] @ V[i] / np.sqrt(rank) + rng.normal(0, 0.4)
            rows.append({"userId": u + 1, "movieId": int(i) + 100,
                         "rating": float(np.clip(np.round(r * 2) / 2, 0.5, 5.0)),
                         "timestamp": 1_000_000 + len(rows)})
    return pd.DataFrame(rows)


def make_movies(ratings: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(ratings["movieId"].unique())
    return pd.DataFrame({"movieId": ids, "title": [f"Synthetic Movie {m} (2000)" for m in ids],
                         "genres": ["Drama|Comedy" if m % 2 else "(no genres listed)" for m in ids]})
