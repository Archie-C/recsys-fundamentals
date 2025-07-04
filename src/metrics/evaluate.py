import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

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


def topk_preds_biased(R_train: csr_matrix,
                     mu: float,
                     bu: np.ndarray,
                     bi: np.ndarray,
                     X: np.ndarray,
                     Y: np.ndarray,
                     k: int):

    m, n = R_train.shape
    scores = X @ Y.T
    scores = mu + bu[:, None] + bi[None, :] + scores
    mask = (R_train > 0).toarray()   # True where user already rated
    scores[mask] = -np.inf
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    topk_sorted = np.vstack([
        topk[u][np.argsort(-scores[u, topk[u]])]
        for u in range(m)
    ])
    return topk_sorted

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

def evaluate_XY(R_train, R_test, X, Y, k=10, biased=False, bu=None, bi=None, mu=0.0):
    if biased:
        preds = topk_preds_biased(R_train, mu, bu, bi, X, Y, k)
    else:
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


def evaluate_content_all_metrics(user_profiles, R_test, items, item_cols, k=10, rating_min=1, rating_max=5):
    # 1) compute profiles
    P = user_profiles[item_cols].values
    X = items[item_cols].values

    # 2) normalize for cosine
    norms = np.linalg.norm(P, axis=1, keepdims=True)   # shape (n_users, 1)
    norms[norms == 0] = 1                               # avoid division by zero
    P_norm = P / norms
    item_norms = np.linalg.norm(X, axis=1, keepdims=True)
    item_norms[item_norms == 0] = 1
    X_norm = X / item_norms

    # 3) predicted rating matrix
    sim = P_norm.dot(X_norm.T)  # in [-1,1]
    preds = (sim + 1) / 2 * (rating_max - rating_min) + rating_min

    # 4) rating metrics
    u_idx, i_idx = R_test.nonzero()
    y_true = R_test.data
    y_pred = preds[u_idx, i_idx]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,

    }
    return metrics