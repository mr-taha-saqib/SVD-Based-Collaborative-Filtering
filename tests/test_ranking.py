"""Top-N ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranking import evaluate_ranking, ndcg_at_k, popularity_scores, top_n


def test_top_n_respects_mask_and_order():
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.2])
    allowed = np.array([True, True, True, False, True])
    assert top_n(scores, allowed, 2).tolist() == [1, 2]
    assert top_n(scores, np.zeros(5, bool), 3).size == 0
    assert len(top_n(scores, allowed, 10)) == 4


def test_ndcg_known_values():
    assert ndcg_at_k(np.array([1, 0, 0]), 1, 3) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([0, 1, 0]), 1, 3) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(np.array([0, 0, 0]), 2, 3) == 0.0
    # two relevant, found at ranks 1 and 3
    exp = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert ndcg_at_k(np.array([1, 0, 1]), 2, 3) == pytest.approx(exp)


def _toy():
    user_idx = {1: 0, 2: 1}
    item_idx = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
    known = pd.DataFrame({"userId": [1, 2], "movieId": [10, 14], "rating": [5.0, 5.0]})
    held = pd.DataFrame({"userId": [1, 1, 2], "movieId": [12, 13, 11], "rating": [5.0, 2.0, 4.0]})
    return user_idx, item_idx, known, held


def test_oracle_scorer_gets_perfect_metrics():
    user_idx, item_idx, known, held = _toy()
    relevant = {0: [2], 1: [1]}

    def oracle(rows):
        s = np.zeros((len(rows), 5))
        for r, row in enumerate(rows):
            s[r, relevant[int(row)]] = 10
        return s

    res = evaluate_ranking("oracle", "val", oracle, held, known, user_idx, item_idx,
                           np.ones(5, bool), k=1, relevance_threshold=4.0)
    assert res.precision == 1.0 and res.recall == 1.0 and res.ndcg == 1.0 and res.hit_rate == 1.0
    assert res.n_users_evaluated == 2


def test_known_items_are_never_recommended():
    user_idx, item_idx, known, held = _toy()

    def loves_known(rows):
        s = np.zeros((len(rows), 5))
        s[:, 0] = 100   # item 10, already rated by user 1
        s[:, 4] = 100   # item 14, already rated by user 2
        return s

    res = evaluate_ranking("x", "val", loves_known, held, known, user_idx, item_idx,
                           np.ones(5, bool), k=1)
    # user 1 cannot get item 10, user 2 cannot get item 14; coverage reflects that
    assert res.catalog_coverage > 0


def test_unrecommendable_relevant_items_count_in_recall():
    user_idx, item_idx, known, _ = _toy()
    held = pd.DataFrame({"userId": [1, 1], "movieId": [12, 999], "rating": [5.0, 5.0]})
    res = evaluate_ranking("pop", "val", popularity_scores(np.array([0, 0, 9, 0, 0])), held, known,
                           user_idx, item_idx, np.ones(5, bool), k=1)
    assert res.recall == pytest.approx(0.5)            # 1 of 2 relevant; unseen 999 still counts
    assert res.n_relevant_not_recommendable == 1
