import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from src.content_based_baseline import compute_content_scores, get_genre_matrix
from src.metrics.evaluate import _ground_truth
from src.metrics.metrics import hr_at_k, item_coverage, ndcg_at_k, precision_at_k, recall_at_k, user_coverage

def compute_cf_scores(mu: float,
                      bu: np.ndarray,
                      bi: np.ndarray,
                      X: np.ndarray,
                      Y: np.ndarray) -> np.ndarray:
    """
    Compute collaborative-filtering score matrix as:
      mu + bu u + bi i + X[u] @ Y[i].T

    Returns dense array of shape (n_users, n_items).
    """
    scores = X @ Y.T
    scores = mu + bu[:, None] + bi[None, :] + scores
    return scores

def topk_hybrid(R_train: csr_matrix,
                cf_scores: np.ndarray,
                content_scores: np.ndarray,
                alpha: float,
                k: int) -> np.ndarray:
    """
    Combine cf_scores and content_scores with weight alpha (0..1):
      hybrid = alpha * cf_scores + (1-alpha) * content_scores
    Mask seen training items and return top-k indices per user.
    """
    hybrid = alpha * cf_scores + (1 - alpha) * content_scores
    # mask seen items so they won't be recommended
    mask = (R_train > 0).toarray()
    hybrid[mask] = -np.inf

    # pick top-k item indices per user
    topk = np.argpartition(-hybrid, k, axis=1)[:, :k]
    # sort those k by descending score
    topk_sorted = np.vstack([
        topk[u][np.argsort(-hybrid[u, topk[u]])]
        for u in range(hybrid.shape[0])
    ])
    return topk_sorted

def evaluate_hybrid(R_train: csr_matrix,
                    R_test: csr_matrix,
                    mu: float,
                    bu: np.ndarray,
                    bi: np.ndarray,
                    X: np.ndarray,
                    Y: np.ndarray,
                    item_meta_df: pd.DataFrame,
                    genre_cols: list,
                    alpha: float = 0.5,
                    k: int = 10) -> dict:
    """
    End-to-end evaluation for hybrid CF + genre-content model.

    Parameters
    ----------
    R_train       : csr_matrix, training interaction matrix
    R_test        : csr_matrix, test interaction matrix
    mu, bu, bi    : bias parameters from ALS training
    X, Y          : latent factor matrices from ALS
    item_meta_df  : DataFrame with columns ['item_id'] + genre_cols
    genre_cols    : list of column names for binary genre features
    alpha         : weight for CF vs content (0=content only,1=CF only)
    k             : number of recommendations per user

    Returns
    -------
    metrics dict with HR, precision, recall, NDCG, user/item coverage
    """
    # 1) build genre matrix
    genre_matrix = get_genre_matrix(item_meta_df, genre_cols)

    # 2) compute score matrices
    cf_scores = compute_cf_scores(mu, bu, bi, X, Y)
    content_scores = compute_content_scores(R_train, genre_matrix)

    # 3) hybrid top-k predictions
    preds = topk_hybrid(R_train, cf_scores, content_scores, alpha, k)

    # 4) ground truth for test
    truth = _ground_truth(R_test)
    n_items = Y.shape[0]

    # 5) compute metrics
    return {
        "hr":            hr_at_k(preds, truth, k),
        "precision":     precision_at_k(preds, truth, k),
        "recall":        recall_at_k(preds, truth, k),
        "ndcg":          ndcg_at_k(preds, truth, k),
        "user_coverage": user_coverage(preds),
        "item_coverage": item_coverage(preds, n_items),
    }
