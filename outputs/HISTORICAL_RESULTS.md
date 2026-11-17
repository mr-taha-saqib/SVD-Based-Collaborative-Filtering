# Historical results (previous evaluation methodology)

The files listed below were produced by the **original** pipeline, before the
methodology correction of November 2026. They are kept unchanged, because
`report/report.tex` and `presentation/slides.tex` reference them, but they must
not be read as results of the corrected method.

| File | Produced by | Status |
|---|---|---|
| `outputs/results_table.csv` | original `src/main.py` (MovieLens latest-small) | historical |
| `outputs/results_ml1m.csv` | original `src/run_ml1m.py` (MovieLens 1M) | historical |
| `outputs/plots/01` to `10_*.png` | original `src/main.py`, `src/run_ml1m.py` | historical |

## What was wrong with the previous method

The original pipeline split the data into **train and test only**, evaluated
every SVD rank `k` on the **test set**, and selected the rank with the lowest
**test** RMSE (`k = 10` on latest-small, `k = 20` on ML-1M). The same test set was
then used to report the final RMSE (0.8812 and 0.8943). Choosing a
hyperparameter on the test set and reporting the score on that same set is test-set
leakage: the reported number is optimistically biased and is not an independent
estimate of generalisation. `README.md` previously described this as
"identified by cross-validation"; no cross-validation was performed.

The SGD-MF baseline inherited the leaked rank (`k = BEST_K`).

## What replaced it

`src/main.py` now uses a per-user stratified **train / validation / test** split
(70 / 10 / 20), selects `k` on **validation RMSE only**, and evaluates the fixed
configuration on **test exactly once**. Results are written to
`outputs/corrected/`; see `outputs/corrected/final_metrics.json` and the
"Results" section of `README.md`. `src/run_ml1m.py` was corrected the same way
and writes to `outputs/corrected/ml1m/`.

The corrected numbers are not directly comparable with the historical ones: the
training partition is smaller (70 % instead of 80 % of each user's ratings), the
split differs, and items that never appear in training are now predicted with an
explicit fallback rather than through an all-empty matrix column.
