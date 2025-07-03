import pytest
import numpy as np

from src.metrics.metrics import (
    hr_at_k,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    user_coverage,
    item_coverage,
)

from src.metrics.evaluate import evaluate


predicted = [
    [1, 2, 3],
    [3, 4, 5],
    []
]
ground_truth = [
    {2, 5},
    {5},
    {1}
]
K = 3
N_ITEMS = 6

def test_hr_at_k():
    # user0 hit (2), user1 hit (5), user2 no hit => 2/3
    assert hr_at_k(predicted, ground_truth, K) == pytest.approx(2/3)


def test_precision_at_k():
    # user0: 1 hit/3, user1: 1/3, user2: 0/3 => avg = (1/3 + 1/3 + 0)/3 = 2/9
    assert precision_at_k(predicted, ground_truth, K) == pytest.approx(2/9)


def test_recall_at_k():
    # user0: 1 hit / 2 relevant = 0.5
    # user1: 1 hit / 1 relevant = 1.0
    # user2: 0 hit / 1 relevant = 0.0
    # avg = (0.5 + 1 + 0)/3 = 1.5/3 = 0.5
    assert recall_at_k(predicted, ground_truth, K) == pytest.approx(0.5)


def test_ndcg_at_k_perfect_and_imperfect():
    # user0: one hit at position 2 => DCG = 1/log2(2+1)
    ndcg0 = (1/np.log2(2+1)) / 1.0
    # user1: one hit at position 3 => DCG = 1/log2(3+1)
    ndcg1 = (1/np.log2(3+1)) / 1.0
    # user2: no hits => 0
    ndcg2 = 0.0
    expected = (ndcg0 + ndcg1 + ndcg2) / 3
    assert ndcg_at_k(predicted, ground_truth, K) == pytest.approx(expected)


def test_user_coverage():
    # two users have non-empty predictions → 2/3
    assert user_coverage(predicted) == pytest.approx(2/3)


def test_item_coverage():
    # predicted items are {1,2,3,4,5} → 5 distinct out of 6
    assert item_coverage(predicted, N_ITEMS) == pytest.approx(5/6)


def test_evaluate_all_metrics():
    ev = evaluate(predicted, ground_truth, K, N_ITEMS)
    # just spot-check keys and approximate values
    assert set(ev) == {"hr", "precision", "recall", "ndcg", "user_coverage", "item_coverage"}
    assert ev["hr"] == pytest.approx(2/3)
    assert ev["precision"] == pytest.approx(2/9)
    assert ev["recall"] == pytest.approx(0.5)
    assert ev["user_coverage"] == pytest.approx(2/3)
    assert ev["item_coverage"] == pytest.approx(5/6)