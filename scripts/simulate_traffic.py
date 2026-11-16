"""Drive a running API with controlled traffic and print /monitoring/summary.

normal : known users sampled uniformly from the model (mirrors the reference profile)
drift  : unknown user ids only (input drift + 100 % cold-start fallback)
feedback: POST /feedback with ratings for random known pairs (delayed labels)

    python scripts/simulate_traffic.py --url http://localhost:8000 --mode normal --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["normal", "drift", "feedback"], default="normal")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-user-id", type=int, default=610, help="largest known userId for normal traffic")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    codes: dict[int, int] = {}
    with httpx.Client(base_url=args.url, timeout=10) as c:
        for _ in range(args.n):
            if args.mode == "normal":
                r = c.get(f"/recommendations/{int(rng.integers(1, args.max_user_id + 1))}", params={"n": 10})
            elif args.mode == "drift":
                r = c.get(f"/recommendations/{int(rng.integers(10**6, 10**7))}", params={"n": 10})
            else:
                body = {"user_id": int(rng.integers(1, args.max_user_id + 1)),
                        "movie_id": int(rng.choice([1, 50, 260, 296, 318, 356, 593, 2571])),
                        "rating": float(rng.choice(np.arange(0.5, 5.01, 0.5)))}
                r = c.post("/feedback", json=body)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
        summary = c.get("/monitoring/summary").json()
    print(json.dumps({"mode": args.mode, "status_codes": codes,
                      "input_drift": summary["input_drift"], "prediction_drift": summary["prediction_drift"],
                      "rates": summary["rates"], "observed_performance": summary["observed_performance"],
                      "latency_ms": summary["latency_ms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
