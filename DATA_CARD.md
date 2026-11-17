# Data card: MovieLens

## Datasets

| | latest-small (primary) | 1M (cross-check) |
|---|---|---|
| Source | https://files.grouplens.org/datasets/movielens/ml-latest-small.zip | https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| Archive checksum | SHA-256 `696d65a3…ebc9e436` (MD5 `31a303aa…` = GroupLens published value) | not pinned |
| Ratings | 100,836 | 1,000,209 |
| Users / movies | 610 / 9,724 | 6,040 / 3,706 |
| Scale | 0.5–5.0 in half stars | 1–5 integers |
| Content fingerprint | `fce385dce15c723ab0f51d166e9d522c2a6ed8dc6ab5c74e2f4edd5ce426fc2f` | recorded in outputs/corrected/ml1m/ |

Files used: `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv`
(movieId, title, genres). Raw data is downloaded at runtime and **not committed**.

## Licence (important for the client)

GroupLens usage licence (from the dataset README): research use only; no
endorsement may be implied; the dataset must be acknowledged; redistribution only
under the same licence; **no commercial or revenue-bearing use without permission from
a faculty member of the GroupLens Research Project**. **Before any commercial deployment the client
must obtain written permission from GroupLens, or replace MovieLens with a commercially
licensed or client-owned rating dataset.** No such permission is held for this delivery,
and none is implied by this repository.

## Validation applied (`validate_ratings`)

Required columns; missing values; integer, positive ids; ratings within the scale
(hard errors); off-step ratings (warning); duplicate (userId, movieId) pairs resolved
by keeping the latest timestamp (reported). On latest-small: 0 invalid rows,
0 duplicates (`outputs/corrected/data_quality.json`).

## Split

Per-user stratified random split, seed 42: train 70,579 / validation 10,093 /
test 20,164 ratings. All 610 users appear in every partition. 408 validation and
817 test movies never occur in training (predicted with a bias fallback). Shared
pairs between any two partitions: 0. Split SHA-256 `c5574af8…` in
`outputs/corrected/split_manifest.json`.

## Known characteristics

Long-tail popularity (many movies with a single rating), active-user skew (every
user has ≥ 20 ratings in latest-small), positive rating skew (mean ≈ 3.5).

## TLS note

`files.grouplens.org` omits its intermediate certificate. `certs/incommon-ovg2c-intermediate.pem`
(public InCommon intermediate, expires 2029-04-07) completes the chain; verification
is never disabled. Behind a TLS-inspecting proxy, point `SSL_CERT_FILE` at the proxy CA.

## Feedback data

`POST /feedback` appends `userId,movieId,rating,timestamp` to
`data/feedback/api_feedback.csv`. All retraining demonstrations in this repository
used **synthetic** feedback (`scripts/generate_synthetic_feedback.py`), clearly labelled.
