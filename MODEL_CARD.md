# Model card: SVD movie recommender

| | |
|---|---|
| Model | Truncated SVD of the bias-centred, zero-imputed user × movie rating matrix |
| Approved version | `svd-k20-fce385dc-c6b9d577` (k = 20, MovieLens latest-small) |
| MLflow run | `4e316c2b14884868b640f5f46af69b96` (experiment `svd-movie-recommender`, alias `champion`) |
| Training data | 70,579 ratings (train partition), dataset SHA-256 `fce385dc…`, split SHA-256 `c5574af8…` |
| Selection | k ∈ {5, 10, 20, 50, 100, 150, 200} by validation RMSE |
| Framework | NumPy / LAPACK; no deep learning; CPU only |

## Intended use

Personalised movie suggestions and rating estimates for users with rating history in
the MovieLens catalogue; research and demonstration. **Not licensed for commercial
use without GroupLens permission** (see DATA_CARD.md).

## Performance (test partition, evaluated once after selection)

| Metric | SVD k=20 | Best baseline |
|---|---|---|
| RMSE | 0.8853 | 0.8943 (user + item bias) |
| MAE | 0.6754 | 0.6834 (user + item bias) |
| Precision@10 | 0.0103 | **0.1108** (most popular) |
| Recall@10 | 0.0022 | **0.0923** (most popular) |
| NDCG@10 | 0.0109 | **0.1470** (most popular) |
| Catalogue coverage@10 | 0.029 | 0.019 (most popular) |

Validation (used for selection): RMSE 0.8729, MAE 0.6691. SGD-MF with the same k
reaches test RMSE 0.8737; it is a comparison model, not served.

## Limitations and risks

* Optimised for rating accuracy, **not** top-N retrieval: offline ranking quality is
  far below a most-popular list. Validate with online experiments before relying on it.
* Unregularised item biases push rarely-rated movies up; the serving policy requires
  at least 5 training ratings per recommended movie (fixed a priori).
* Popularity bias in the fallback; no diversity or novelty controls.
* Random per-user split, not temporal: results can be optimistic for future behaviour.
* Cold-start users get non-personalised results; new movies need ratings first.
* Predicted ratings are estimates with typical error ≈ 0.68 stars (MAE).
* Licence: trained on MovieLens, research use only (`docs/DATASET_LICENSING.md`).

## Alternative model: MovieTweetings (separate registry, not the delivered model)

Same algorithm and methodology on MovieTweetings (20-core, ratings / 2): k = 5 selected on
validation (lower edge of the grid), test RMSE 0.6930 / MAE 0.5122 vs user + item bias
0.6986 / 0.5170; NDCG@10 0.0039 vs most popular 0.0568. Details and caveats:
`docs/MOVIETWEETINGS_EVALUATION.md`. Not cleared for commercial use.

## Ethical considerations

MovieLens ratings are pseudonymous. The API logs only salted hashes of user ids in
the monitoring stream; feedback stored for retraining keeps raw ids (required to
retrain) and should be protected accordingly.

## Historical model

The earlier study reported k = 10, RMSE 0.8812, selected on the test set. That number
is optimistically biased; see outputs/HISTORICAL_RESULTS.md.
