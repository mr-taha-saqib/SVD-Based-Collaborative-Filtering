# Retraining and model promotion

Implemented in `src/retrain.py`; executed end to end on a copy of the registry
(`docs/evidence/06_retraining_e2e.log`, reports in `docs/evidence/retraining_reports/`).

## Interface

CSV of **new** ratings, columns `userId,movieId,rating[,timestamp]`. The API's
`POST /feedback` writes exactly this format to `data/feedback/api_feedback.csv`.

```bash
python src/retrain.py --feedback FILE.csv --dry-run    # everything except registry changes
python src/retrain.py --feedback FILE.csv              # promote only if the gate passes
python src/retrain.py --feedback FILE.csv --force      # re-run even if already processed
python src/retrain.py --rollback --reason "..."        # re-approve the previous model
```

## Steps

1. Load the approved version; fail if none.
2. Reload the base data; its fingerprint must equal the approved model's (else abort).
3. Re-create the base split; its checksum must match (else abort). The **test
   partition is not used anywhere below** (enforced by
   `tests/test_retrain.py::test_retraining_never_reads_the_test_partition`).
4. Validate feedback: invalid rows dropped and counted; duplicates resolved; pairs that
   already exist in base data or earlier promoted feedback are dropped (they would
   alter train/validation/test).
5. Minimum requirements: ≥ 100 new ratings, ≥ 5 users, more than one rating value.
6. New ratings are split deterministically (salted hash of user/movie) into 80 %
   feedback-train and 20 % feedback-validation.
7. Candidate: SVD on base-train + feedback-train; k re-selected on validation
   (base-validation + feedback-validation).
8. The approved model is scored on the same validation data.
9. Gate (`config.yaml` → `retraining.promotion`), all on validation:
   * validation RMSE ≤ approved − `min_rmse_improvement` (0.0);
   * RMSE on the original validation partition not worse by more than 0.002
     (protects existing users against poisoned or skewed feedback);
   * NDCG@10 not lower by more than 0.01;
   * all candidate metrics finite.
10. The candidate is registered either way; promoted only if every criterion passes and
    this is not a dry run. The previous version stays in `models/versions/` and in the
    registry history. Every decision is stored with its evidence in `registry.json`
    and in MLflow (aliases `champion`, `rejected-latest`).
11. Reports: `reports/retraining/retraining-<timestamp>.{json,md}`.

Idempotence: a marker keyed on (approved version, feedback fingerprint, retraining
config) prevents re-processing the same file against the same model.

## Executed scenarios (synthetic feedback, isolated registry copy)

| # | Feedback | Outcome |
|---|---|---|
| 1 | 40 ratings | `rejected_data_requirements` (40 < 100), nothing trained |
| 2 | 3,009 rows incl. malformed ones, `--dry-run` | 4 invalid rows dropped, 5 duplicates resolved; gate failed; nothing changed |
| 3 | 3,000 random ratings | candidate `…baf57cab` **rejected** (validation RMSE worse) |
| 4 | 3,000 ratings from 47 new synthetic users | candidate `…9180e770` **promoted**: validation RMSE 0.8671 → 0.8604, base-validation 0.8729 → 0.8733 (within tolerance) |
| 5 | same file again | `rejected_data_requirements` (all pairs already known) |
| 7 | rollback | approved pointer returned to `svd-k20-fce385dc-c6b9d577` |

**Synthetic data only.** These runs demonstrate the mechanics. Real retraining needs
genuinely observed ratings; the delivered `models/` registry was not modified by
these demonstrations.
