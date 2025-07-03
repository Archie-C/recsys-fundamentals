import numpy as np
from scipy import sparse
from src.metrics.metrics import hr_at_k, item_coverage, ndcg_at_k, precision_at_k, recall_at_k, user_coverage


def topk_preds(R_train, X, Y, k):
    m, n = R_train.shape
    Yt = Y.T
    preds = []
    for u in range(m):
        # user‐scores and mask training interactions
        scores = X[u] @ Yt 
        mask   = R_train[u].indices
        scores[mask] = -np.inf
        # get top‐k
        idx = np.argpartition(-scores, k)[:k]
        # sort those k
        topk = idx[np.argsort(-scores[idx])]
        preds.append(topk.tolist())
    return preds

def _ground_truth(R_test):
    m, _ = R_test.shape
    out = []
    coo = R_test.tocoo()
    for u in range(m):
        out.append(set(coo.col[coo.row == u].tolist()))
    return out

def evaluate(predicted, ground_truth, k, n_items):
    return {
        "hr":              hr_at_k(predicted, ground_truth, k),
        "precision":       precision_at_k(predicted, ground_truth, k),
        "recall":          recall_at_k(predicted, ground_truth, k),
        "ndcg":            ndcg_at_k(predicted, ground_truth, k),
        "user_coverage":   user_coverage(predicted),
        "item_coverage":   item_coverage(predicted, n_items),
    }

def evaluate_XY(R_train, R_test, X, Y, k=10):
    preds = topk_preds(R_train, X, Y, k)
    truth = _ground_truth(R_test)
    n_items = Y.shape[0]

    return {
        "hr":            hr_at_k(preds, truth, k),
        "precision":     precision_at_k(preds, truth, k),
        "recall":        recall_at_k(preds, truth, k),
        "ndcg":          ndcg_at_k(preds, truth, k),
        "user_coverage": user_coverage(preds),
        "item_coverage": item_coverage(preds, n_items),
    }