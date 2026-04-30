# SVD-Based Collaborative Filtering

A mathematically rigorous recommendation system built on **Singular Value Decomposition**, validated on two MovieLens datasets (100K and 1M ratings).

## Overview

This project applies the **Eckart-Young-Mirsky theorem** to show that truncated SVD is the *provably optimal* rank-k matrix approximation. The full pipeline covers data loading, bias decomposition, SVD, rank selection, baseline comparison, and cross-dataset validation.

**Key results:**
| Model | RMSE | vs Global Mean |
|---|---|---|
| Global Mean | 1.0447 | — |
| User Mean | 0.9435 | −9.7% |
| User+Item Bias | 0.8930 | −14.6% |
| **SVD k=10** | **0.8812** | **−15.6%** |
| SGD-MF k=10 | 0.8710 | −16.6% |

Cross-dataset validation on ML-1M (1M ratings, 10× scale): RMSE = 0.8943, best k = 20. Model hierarchy preserved.

## Project Structure

```
SVD-Based-Collaborative-Filtering/
├── src/
│   ├── main.py             # Full ML-small pipeline
│   ├── run_ml1m.py         # ML-1M cross-dataset validation
│   ├── data_loader.py      # Download, split, matrix construction, bias decomposition
│   ├── svd_model.py        # SVD, Eckart-Young bound, prediction
│   ├── evaluation.py       # RMSE, MAE, baselines
│   └── visualizations.py   # All 10 publication-quality figures
├── tests/
│   └── test_pipeline.py    # 48 unit tests (all pass in 1.39s)
├── outputs/
│   ├── plots/              # 10 generated figures
│   ├── results_table.csv   # ML-small results
│   └── results_ml1m.csv    # ML-1M results
├── report/
│   └── report.tex          # Full LaTeX academic report
├── presentation/
│   └── slides.tex          # Beamer presentation slides
├── config.yaml             # All hyperparameters
└── requirements.txt
```

> **Note:** Raw dataset files are not included. They are downloaded automatically on first run from [GroupLens](https://grouplens.org/datasets/movielens/).

## Setup

```bash
pip install -r requirements.txt
```

## Run

**ML-small pipeline (full):**
```bash
cd src
python main.py
```

**ML-1M cross-dataset validation:**
```bash
cd src
python run_ml1m.py
```

**Test suite (48 tests):**
```bash
pytest tests/
```

## Mathematical Foundation

Rating prediction formula:

```
r̂_ui = μ + b_u + b_i + Σ_l σ_l · U_k[u,l] · V_k[i,l]
```

Where:
- `μ` = global mean, `b_u` = user bias, `b_i` = item bias (bias decomposition)
- `U_k, Σ_k, V_k` = truncated SVD of the zero-filled centered matrix

**Eckart-Young theorem** guarantees that for any rank k, the SVD truncation minimises the Frobenius-norm approximation error — no other rank-k matrix achieves lower error.

## Results

### Rank Selection (k-sweep, ML-small)

| k | RMSE | MAE | EV% |
|---|---|---|---|
| 5 | 0.8829 | 0.6725 | 9% |
| **10** | **0.8812** | **0.6708** | **15.5%** |
| 20 | 0.8825 | 0.6721 | 25% |
| 50 | 0.8852 | 0.6740 | 45% |
| 100 | 0.8897 | 0.6784 | 64% |

Optimal rank k=10 identified by cross-validation. RMSE rises beyond k=10 due to overfitting to zero-imputed entries (bias-variance tradeoff).

### Cross-Dataset Validation

| | ML-small | ML-1M |
|---|---|---|
| Users | 610 | 6,040 |
| Items | 9,724 | 3,706 |
| Ratings | 100,836 | 1,000,209 |
| Density | 1.36% | 4.47% |
| Best k | 10 | 20 |
| RMSE (SVD) | 0.8812 | 0.8943 |

## Test Suite

48 tests across 8 categories — all pass in 1.39s.

| Category | Tests |
|---|---|
| Data Validation | 8 |
| Model Validation | 12 |
| Metrics Validation | 9 |
| Bias Decomposition | 4 |
| Reproducibility | 4 |
| Edge Cases | 5 |
| Performance | 3 |
| Results Consistency | 3 |

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
scipy
pytest
```

## Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) — 100,836 ratings by 610 users on 9,724 movies.  
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1,000,209 ratings by 6,040 users on 3,706 movies.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4).
