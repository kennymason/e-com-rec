import math

import pytest

from shoprec.eval.metrics import ap_at_k, bootstrap_ci, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4) == pytest.approx(2 / 3)
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k([], {1}, k=5) == 0.0
    assert recall_at_k([1], set(), k=5) == 0.0


def test_ap_matches_kaggle_definition():
    # hits at ranks 1 and 3: (1/1 + 2/3) / min(2, 12)
    assert ap_at_k([1, 9, 2], {1, 2}, k=12) == pytest.approx((1 + 2 / 3) / 2)
    # more relevant items than k: the denominator is k
    assert ap_at_k([1, 2], {1, 2, 3}, k=2) == pytest.approx(1.0)
    assert ap_at_k([9, 8], {1}, k=12) == 0.0


def test_ndcg():
    assert ndcg_at_k([1, 2], {1, 2}, k=12) == pytest.approx(1.0)
    assert ndcg_at_k([9, 1], {1}, k=12) == pytest.approx(1 / math.log2(3))
    # one of two relevant items, at rank 1
    assert ndcg_at_k([1, 9], {1, 2}, k=12) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([9, 1], {1}, k=1) == 0.0


def test_duplicate_recommendations_count_once():
    assert recall_at_k([1, 1, 1], {1, 2}, k=3) == pytest.approx(0.5)
    assert ap_at_k([1, 1], {1, 2}, k=12) == pytest.approx(0.5)
    assert ndcg_at_k([1, 1], {1}, k=12) == pytest.approx(1.0)


def test_bootstrap_ci():
    values = [0.0, 1.0] * 200
    lo, hi = bootstrap_ci(values, n_boot=500)
    assert lo < 0.5 < hi
    assert hi - lo < 0.2
    assert bootstrap_ci(values, n_boot=500) == (lo, hi)  # deterministic for a given seed
    assert bootstrap_ci([0.3] * 10) == pytest.approx((0.3, 0.3))
