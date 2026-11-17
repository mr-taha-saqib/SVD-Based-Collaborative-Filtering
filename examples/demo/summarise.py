"""Pretty-print API responses for run_demo.sh (reads JSON from stdin)."""
import json
import sys

mode = sys.argv[1]
d = json.load(sys.stdin)
if mode == "info":
    keys = ("model_version", "algorithm", "k", "n_users", "n_items", "serving_policy", "limits")
    print(json.dumps({k: d.get(k) for k in keys if k in d}, indent=2, ensure_ascii=False))
elif mode == "recs":
    print(f"user {d['user_id']}  strategy {d['strategy']}  known_user {d['known_user']}  "
          f"returned {d['n_returned']}/{d['n_requested']}")
    for it in d["items"]:
        pr = "" if it["predicted_rating"] is None else f"  predicted {it['predicted_rating']:.2f}"
        print(f"  {it['rank']:>2}. {it['title'] or it['movie_id']}{pr}")
    if d.get("unknown_excluded_movie_ids"):
        print("  unknown excluded ids:", d["unknown_excluded_movie_ids"])
elif mode == "summary":
    keys = ("model_version", "requests_in_window", "rates", "input_drift", "prediction_drift", "observed_performance")
    out = {}
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict) and k.endswith("drift"):
            v = {x: v.get(x) for x in ("status", "psi") if x in v}
        out[k] = v
    print(json.dumps(out, indent=2, ensure_ascii=False))
