"""
retrain.py
==========
Controlled retraining with an explicit promotion gate.

    python src/retrain.py --feedback data/feedback/new_ratings.csv            # train, gate, promote if passed
    python src/retrain.py --feedback new.csv --dry-run                        # evaluate only, change nothing
    python src/retrain.py --rollback --reason "regression reported"           # re-approve previous model

Interface: a CSV of NEW ratings with columns ``userId,movieId,rating[,timestamp]``
(the API's ``POST /feedback`` appends to ``data/feedback/api_feedback.csv`` in
exactly this format).

Workflow
--------
 1. Load the APPROVED model and its metadata (fails if none is approved).
 2. Reload the base dataset and verify its fingerprint matches the approved model.
 3. Re-create the frozen base split and verify its checksum matches.
 4. Validate the feedback: schema, ranges, ids, duplicates; drop pairs that
    already exist in the base data (they would alter train/validation/test).
 5. Check minimum data requirements and rating coverage.
 6. Split NEW ratings deterministically (hash of user/movie) into
    feedback-train and feedback-validation.
 7. Train a candidate on base-train + feedback-train; choose k on validation
    (base-validation + feedback-validation).
 8. Evaluate the approved model on the SAME validation data.
 9. Apply the promotion criteria (config ``retraining.promotion``).
10. Register the candidate, log the MLflow run, record the decision.
11. Promote only if every criterion passes (and not --dry-run). The previous
    approved model stays in the registry for rollback.
12. Write a JSON + Markdown retraining report.

The base TEST partition is never read during retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (  # noqa: E402
    DataValidationError,
    dataset_fingerprint,
    load_movies,
    load_ratings_from_config,
    train_val_test_split,
    validate_ratings,
)
from evaluation import evaluate_pairs  # noqa: E402
from ranking import evaluate_ranking  # noqa: E402
from recommender import Recommender, build_artifact  # noqa: E402
from registry import ModelRegistry, RegistryError  # noqa: E402
from settings import get, load_config, resolve_path  # noqa: E402
from tracking import Tracker, git_commit  # noqa: E402
from training import candidate_mask, decompose, select_rank  # noqa: E402

logger = logging.getLogger("retrain")


class RetrainingError(RuntimeError):
    """Raised when retraining cannot proceed (bad data, missing approved model...)."""


@dataclass
class GateResult:
    passed: bool
    criteria: list[dict[str, Any]] = field(default_factory=list)


def promotion_gate(candidate: dict[str, float], approved: dict[str, float], rules: dict[str, Any],
                   ndcg_key: str) -> GateResult:
    """Pure function: compare validation metrics and apply explicit criteria.

    Every metric here comes from the VALIDATION data; the test partition is not an input.
    """
    min_imp = float(rules.get("min_rmse_improvement", 0.0))
    max_base_deg = float(rules.get("max_base_rmse_degradation", 0.002))
    max_ndcg_drop = float(rules.get("max_ndcg_drop", 0.01))
    crit = []

    def add(name: str, ok: bool, detail: str) -> None:
        crit.append({"criterion": name, "passed": bool(ok), "detail": detail})

    finite = all(np.isfinite(candidate.get(k, np.nan)) for k in ("rmse", "base_rmse", ndcg_key))
    add("candidate_metrics_finite", finite, "all candidate validation metrics are finite numbers")
    add("validation_rmse_improves",
        finite and candidate["rmse"] <= approved["rmse"] - min_imp,
        f"candidate {candidate.get('rmse'):.4f} <= approved {approved['rmse']:.4f} - {min_imp}")
    add("base_validation_rmse_not_degraded",
        finite and candidate["base_rmse"] <= approved["base_rmse"] + max_base_deg,
        f"candidate {candidate.get('base_rmse'):.4f} <= approved {approved['base_rmse']:.4f} + {max_base_deg}")
    add("ndcg_not_dropped",
        finite and candidate[ndcg_key] >= approved[ndcg_key] - max_ndcg_drop,
        f"candidate {candidate.get(ndcg_key):.4f} >= approved {approved[ndcg_key]:.4f} - {max_ndcg_drop}")
    return GateResult(passed=all(c["passed"] for c in crit), criteria=crit)


def _hash_fraction(user: int, movie: int, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{user}:{movie}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def prepare_feedback(feedback: pd.DataFrame, base: pd.DataFrame, cfg: dict[str, Any],
                     known_users: set[int], known_items: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate new ratings and split them into feedback-train / feedback-validation."""
    rcfg = cfg.get("retraining", {})
    scale = tuple(get(cfg, "dataset.rating_scale", [0.5, 5.0]))
    try:
        clean, rep = validate_ratings(feedback, rating_scale=scale, drop_invalid=True)
    except DataValidationError as exc:
        raise RetrainingError(f"Feedback rejected: {exc}") from exc
    report = {"validation": rep.to_dict()}
    base_pairs = set(zip(base["userId"].tolist(), base["movieId"].tolist()))
    is_dup = np.array([(u, m) in base_pairs for u, m in zip(clean["userId"], clean["movieId"])], dtype=bool)
    report["dropped_existing_pairs"] = int(is_dup.sum())
    new = clean.loc[~is_dup].reset_index(drop=True)
    report["n_new_ratings"] = int(len(new))
    report["n_users"] = int(new["userId"].nunique()) if len(new) else 0
    report["n_items"] = int(new["movieId"].nunique()) if len(new) else 0
    report["share_known_users"] = float(new["userId"].isin(known_users).mean()) if len(new) else 0.0
    report["share_known_items"] = float(new["movieId"].isin(known_items).mean()) if len(new) else 0.0
    report["rating_mean"] = float(new["rating"].mean()) if len(new) else None
    report["rating_histogram"] = {str(k): int(v) for k, v in new["rating"].value_counts().sort_index().items()}

    min_new = int(rcfg.get("min_new_ratings", 100))
    min_users = int(rcfg.get("min_users", 5))
    problems = []
    if report["n_new_ratings"] < min_new:
        problems.append(f"{report['n_new_ratings']} new ratings < minimum {min_new}")
    if report["n_users"] < min_users:
        problems.append(f"{report['n_users']} users < minimum {min_users}")
    if len(new) and new["rating"].nunique() < 2:
        problems.append("all new ratings have the same value (no signal)")
    report["requirements_passed"] = not problems
    report["requirement_failures"] = problems
    if problems:
        return new.iloc[0:0], new.iloc[0:0], report

    frac = float(rcfg.get("feedback_validation_fraction", 0.2))
    salt = str(get(cfg, "split.random_state", 42))
    to_val = np.array([_hash_fraction(u, m, salt) < frac for u, m in zip(new["userId"], new["movieId"])])
    fb_train, fb_val = new.loc[~to_val].reset_index(drop=True), new.loc[to_val].reset_index(drop=True)
    report["n_feedback_train"] = int(len(fb_train))
    report["n_feedback_validation"] = int(len(fb_val))
    return fb_train, fb_val, report


def _metrics_on(model, dec_user_idx, dec_item_idx, val_all, base_val, known, item_counts,
                rank_cfg) -> dict[str, float]:
    k = int(rank_cfg.get("k", 10))
    all_res = evaluate_pairs("eval", model, val_all, dec_user_idx, dec_item_idx)
    base_res = evaluate_pairs("eval-base", model, base_val, dec_user_idx, dec_item_idx)
    rr = evaluate_ranking("eval", "validation", model.score_users, val_all, known, dec_user_idx,
                          dec_item_idx, candidate_mask(item_counts, rank_cfg.get("min_item_ratings", 5)), k=k,
                          relevance_threshold=float(rank_cfg.get("relevance_threshold", 4.0)))
    return {"rmse": all_res.rmse, "mae": all_res.mae, "base_rmse": base_res.rmse,
            f"ndcg@{k}": rr.ndcg, f"precision@{k}": rr.precision, f"recall@{k}": rr.recall,
            "catalog_coverage": rr.catalog_coverage, "n_validation": all_res.n_samples}


def run_retraining(cfg: dict[str, Any], feedback: pd.DataFrame, dry_run: bool = False,
                   base_ratings: pd.DataFrame | None = None, movies: pd.DataFrame | None = None,
                   force: bool = False) -> dict[str, Any]:
    """Execute the workflow. Returns the decision report (also written to disk)."""
    reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "models")))
    try:
        approved_id = reg.approved_version()
        approved_dir = reg.approved_dir()
    except RegistryError as exc:
        raise RetrainingError(f"No approved model to compare against: {exc}") from exc
    approved_meta = reg.metadata(approved_id)
    rank_cfg = cfg.get("ranking", {})
    rk = int(rank_cfg.get("k", 10))
    ndcg_key = f"ndcg@{rk}"
    report_dir = resolve_path(get(cfg, "retraining.report_dir", "reports/retraining"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    # 2-3. base data + split integrity
    if base_ratings is None:
        base_ratings, _ = load_ratings_from_config(cfg)
    else:
        base_ratings, _ = validate_ratings(base_ratings, rating_scale=tuple(get(cfg, "dataset.rating_scale", [0.5, 5.0])))
    data_sha = dataset_fingerprint(base_ratings)
    expected_sha = approved_meta.get("training_data", {}).get("dataset_sha256")
    base_source_sha = approved_meta.get("training_data", {}).get("base_dataset_sha256", expected_sha)
    if base_source_sha and data_sha != base_source_sha:
        raise RetrainingError(f"Base dataset fingerprint {data_sha[:12]} does not match the approved model's "
                              f"{base_source_sha[:12]}; refusing to compare on different data.")
    split_cfg = cfg.get("split", {})
    split = train_val_test_split(base_ratings, float(split_cfg.get("validation_fraction", 0.1)),
                                 float(split_cfg.get("test_fraction", 0.2)), int(split_cfg.get("random_state", 42)))
    exp_split = approved_meta.get("training_data", {}).get("base_split_sha256",
                                                           approved_meta.get("training_data", {}).get("split_sha256"))
    if exp_split and split.manifest["split_sha256"] != exp_split:
        raise RetrainingError("Re-created base split does not match the approved model's split checksum.")
    base_train, base_val = split.train, split.validation          # split.test is NOT used below

    # Earlier promoted feedback that the approved model was trained on
    prev_fb_train = _load_feedback_history(approved_dir, "feedback_train.csv")
    prev_fb_val = _load_feedback_history(approved_dir, "feedback_validation.csv")
    known_all = pd.concat([base_ratings, prev_fb_train, prev_fb_val], ignore_index=True)

    # 4-6. feedback
    fb_sha = dataset_fingerprint(validate_ratings(feedback, drop_invalid=True)[0]) if len(feedback) else "empty"
    candidate_key = hashlib.sha256(f"{approved_id}|{fb_sha}|{json.dumps(cfg.get('retraining'), sort_keys=True)}"
                                   .encode()).hexdigest()[:12]
    marker = report_dir / f"done-{candidate_key}.json"
    if marker.exists() and not force and not dry_run:
        logger.info("Feedback %s against approved %s was already processed; returning previous report "
                    "(use --force to re-run)", fb_sha[:12], approved_id)
        return json.loads(marker.read_text(encoding="utf-8"))

    rec_approved = Recommender.load(approved_dir)
    fb_train, fb_val, fb_report = prepare_feedback(
        feedback, known_all, cfg, set(rec_approved.user_index), set(rec_approved.item_index))
    report: dict[str, Any] = {"timestamp": stamp, "approved_version": approved_id, "dry_run": dry_run,
                              "candidate_key": candidate_key, "feedback_sha256": fb_sha,
                              "base_dataset_sha256": data_sha, "feedback": fb_report,
                              "test_partition_used": False, "git_commit": git_commit()}
    if not fb_report["requirements_passed"]:
        report["decision"] = "rejected_data_requirements"
        report["promoted"] = False
        report["approved_after"] = reg.approved_version()
        logger.warning("Retraining aborted: %s", fb_report["requirement_failures"])
        _write_reports(report, report_dir, stamp)
        return report

    train_all = pd.concat([base_train, prev_fb_train, fb_train], ignore_index=True)
    val_all = pd.concat([base_val, prev_fb_val, fb_val], ignore_index=True)

    # 7. candidate
    k_values = [int(k) for k in get(cfg, "model.k_values", [5, 10, 20, 50])]
    dec = decompose(train_all, max(k_values), get(cfg, "preprocessing.imputation_strategy", "zero"),
                    random_state=int(split_cfg.get("random_state", 42)))
    best_k, sweep = select_rank(dec, k_values, val_all)
    cand_model = dec.model(best_k)
    cand_metrics = _metrics_on(cand_model, dec.user_idx, dec.item_idx, val_all, base_val, train_all,
                               dec.item_counts, rank_cfg)

    # 8. approved model on the same validation data (it knows only its own training data)
    approved_known = pd.concat([base_train, prev_fb_train], ignore_index=True)
    appr_metrics = _metrics_on(rec_approved.model, rec_approved.user_index, rec_approved.item_index,
                               val_all, base_val, approved_known, rec_approved.item_counts, rank_cfg)

    # 9. gate
    gate = promotion_gate(cand_metrics, appr_metrics, cfg.get("retraining", {}).get("promotion", {}), ndcg_key)
    report.update({"candidate": {"selected_k": best_k, "validation": cand_metrics,
                                 "k_sweep": sweep.to_dict(orient="records")},
                   "approved": {"version": approved_id, "validation": appr_metrics},
                   "gate": asdict(gate)})

    # 10-11. register / track / promote
    tracker = Tracker(cfg)
    with tracker.run(run_name=f"retrain-{stamp}", tags={"stage": "retraining", "dry_run": str(dry_run),
                                                       "approved_version": approved_id,
                                                       "dataset_sha256": data_sha,
                                                       "feedback_sha256": fb_sha}):
        tracker.log_params({"selected_k": best_k, "k_values": k_values, "n_feedback_train": len(fb_train),
                            "n_feedback_validation": len(fb_val), "approved_version": approved_id,
                            "promotion_rules": json.dumps(get(cfg, "retraining.promotion", {}))})
        tracker.log_metrics({f"candidate_val_{k.replace('@', '_at_')}": v for k, v in cand_metrics.items()})
        tracker.log_metrics({f"approved_val_{k.replace('@', '_at_')}": v for k, v in appr_metrics.items()})
        tracker.log_metrics({"gate_passed": float(gate.passed)})
        version_id = None
        if not dry_run:
            if movies is None:
                movies = load_movies(resolve_path(get(cfg, "dataset.movies_path", "data/ml-latest-small/movies.csv")))
            seen = pd.concat([known_all, fb_train, fb_val], ignore_index=True)
            with tempfile.TemporaryDirectory() as tmp:
                meta = {"model_name": get(cfg, "registry.model_name", "svd-movie-recommender"),
                        "algorithm": "truncated SVD with additive bias decomposition", "k": best_k,
                        "created_at": stamp, "parent_version": approved_id,
                        "training_data": {
                            "base_dataset_sha256": data_sha, "dataset_sha256": data_sha,
                            "base_split_sha256": split.manifest["split_sha256"],
                            "feedback_sha256": fb_sha,
                            "trained_on": "base train partition + feedback-train",
                            "n_train_ratings": int(len(train_all))},
                        "metrics": {"validation": cand_metrics,
                                    "test": "not evaluated during retraining (test partition untouched)"},
                        "promotion_gate": asdict(gate),
                        "config": {k: cfg.get(k) for k in ("split", "model", "ranking", "preprocessing")},
                        "git_commit": git_commit(), "mlflow_run_id": tracker.run_id}
                content_sha = build_artifact(tmp, cand_model, dec.user_ids, dec.item_ids, train_all, seen, movies,
                                             meta, min_item_ratings=int(rank_cfg.get("min_item_ratings", 5)),
                                             reference_top_k=rk)
                pd.concat([prev_fb_train, fb_train]).to_csv(Path(tmp) / "feedback_train.csv", index=False)
                pd.concat([prev_fb_val, fb_val]).to_csv(Path(tmp) / "feedback_validation.csv", index=False)
                version_id = f"svd-k{best_k}-{data_sha[:8]}-{content_sha[:8]}"
                meta_written = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
                meta_written["version_id"] = version_id
                vdir = reg.register(tmp, version_id, meta_written)
            tracker.log_artifacts(vdir, artifact_path="model")
            mlflow_version = tracker.register_model(version_id)
            decision = {"reason": "promotion gate passed" if gate.passed else "promotion gate failed",
                        "type": "automated_gate", "criteria": gate.criteria,
                        "candidate_validation": cand_metrics, "approved_validation": appr_metrics,
                        "mlflow_run_id": tracker.run_id}
            if gate.passed:
                reg.approve(version_id, decision)
                tracker.set_alias("champion", mlflow_version)
                report["decision"] = "promoted"
            else:
                reg.record_rejection(version_id, decision)
                tracker.set_alias("rejected-latest", mlflow_version)
                report["decision"] = "rejected_by_gate"
            report["candidate_version"] = version_id
            report["mlflow_model_version"] = mlflow_version
        else:
            report["decision"] = "dry_run_gate_passed" if gate.passed else "dry_run_gate_failed"
        tracker.set_tags({"decision": report["decision"]})
    report["promoted"] = report["decision"] == "promoted"
    report["mlflow_run_id"] = tracker.run_id
    report["approved_after"] = reg.approved_version()
    _write_reports(report, report_dir, stamp)
    if not dry_run:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Retraining decision: %s (approved now: %s)", report["decision"], report["approved_after"])
    return report


def _load_feedback_history(version_dir: Path, name: str) -> pd.DataFrame:
    p = version_dir / name
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"]).astype(
        {"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"})


def _write_reports(report: dict[str, Any], report_dir: Path, stamp: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    jpath = report_dir / f"retraining-{stamp}.json"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [f"# Retraining report {stamp}", "",
             f"* Approved model at start: `{report.get('approved_version')}`",
             f"* Dry run: {report.get('dry_run')}",
             f"* Decision: **{report.get('decision')}**",
             f"* Approved model after run: `{report.get('approved_after', report.get('approved_version'))}`",
             f"* Candidate version: `{report.get('candidate_version', 'n/a')}`",
             f"* MLflow run: `{report.get('mlflow_run_id')}`",
             "* Test partition used: **no**", "",
             "## Feedback data", "",
             f"* New ratings accepted: {report['feedback'].get('n_new_ratings')}",
             f"* Dropped (pair already known): {report['feedback'].get('dropped_existing_pairs')}",
             f"* Requirement failures: {report['feedback'].get('requirement_failures') or 'none'}", ""]
    if "gate" in report:
        lines += ["## Validation comparison", "", "| metric | approved | candidate |", "|---|---|---|"]
        for k, v in report["candidate"]["validation"].items():
            a = report["approved"]["validation"].get(k)
            lines.append(f"| {k} | {a:.4f} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {a} | {v} |")
        lines += ["", "## Promotion criteria", "", "| criterion | passed | detail |", "|---|---|---|"]
        lines += [f"| {c['criterion']} | {c['passed']} | {c['detail']} |" for c in report["gate"]["criteria"]]
    (report_dir / f"retraining-{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Retraining report written to %s", jpath)


def main(argv: list[str] | None = None) -> int:
    from logging_utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback", help="CSV of new ratings (userId,movieId,rating[,timestamp])")
    ap.add_argument("--dry-run", action="store_true", help="validate, train and evaluate; change nothing")
    ap.add_argument("--force", action="store_true", help="re-run even if this feedback was already processed")
    ap.add_argument("--rollback", action="store_true", help="re-approve the previous approved model")
    ap.add_argument("--reason", default="manual rollback")
    ap.add_argument("--config")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.no_mlflow:
        cfg.setdefault("mlflow", {})["enabled"] = False
    setup_logging(get(cfg, "output.log_level", "INFO"))
    if args.rollback:
        v = ModelRegistry(resolve_path(get(cfg, "registry.dir", "models"))).rollback(args.reason)
        print(json.dumps({"rolled_back_to": v}))
        return 0
    if not args.feedback:
        ap.error("--feedback is required unless --rollback is given")
    path = Path(args.feedback)
    if not path.exists():
        logger.error("Feedback file not found: %s", path)
        return 2
    try:
        rep = run_retraining(cfg, pd.read_csv(path), dry_run=args.dry_run, force=args.force)
    except RetrainingError as exc:
        logger.error("Retraining failed: %s", exc)
        return 1
    print(json.dumps({k: rep.get(k) for k in ("decision", "promoted", "approved_version",
                                               "candidate_version", "approved_after", "mlflow_run_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
