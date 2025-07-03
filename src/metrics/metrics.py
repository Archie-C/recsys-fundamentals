import numpy as np

def hr_at_k(predicted, actual, k=10):
    hits=0
    n_users = len(predicted)
    for u in range(n_users):
        pred_u = predicted[u][:k]
        if actual[u] & set(pred_u):
            hits += 1
    return hits / n_users if n_users > 0 else 0.0

def precision_at_k(predicted, actual, k=10):
    total = 0
    n_users = len(predicted)
    for u in range(n_users):
        pred_u = predicted[u][:k]
        total += len(set(pred_u) & actual[u])
    return total / (n_users * k) if n_users > 0 else 0.0

def recall_at_k(predicted, actual, k=10):
    total_recall = 0.0
    n_users = len(predicted)
    
    for u in range(n_users):
        gt = actual[u]
        if not gt:
            continue
        pred_u = predicted[u][:k]
        total_recall += len(set(pred_u) & gt) / len(gt)
    
    return total_recall / n_users if n_users > 0 else 0.0

def dcg_at_k(rels) -> float:

    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels))


def ndcg_at_k(predicted, ground_truth, k) -> float:
    total = 0.0
    n_users = len(predicted)
    for u in range(n_users):
        pred_u = predicted[u][:k]
        # binary relevance vector
        rels = [1 if i in ground_truth[u] else 0 for i in pred_u]
        dcg = dcg_at_k(rels)
        # ideal (all 1s then 0s)
        ideal_rels = sorted(rels, reverse=True)
        idcg = dcg_at_k(ideal_rels)
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / n_users

def user_coverage(predicted):
    n_users = len(predicted)
    non_empty_users = sum(1 for p in predicted if len(p) > 0)
    return non_empty_users / n_users if n_users > 0 else 0.0

def item_coverage(predicted, n_items):
    all_recs = set()
    for p in predicted:
        all_recs.update(i for i in p if not np.isnan(i))
    return len(all_recs) / n_items if n_items > 0 else 0.0