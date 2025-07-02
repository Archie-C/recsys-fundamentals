import numpy as np

def cosine_similarity(u, v):
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) == 0:
        return 0
    u_masked = u[mask]
    v_masked = v[mask]
    a, b = np.linalg.norm(u_masked), np.linalg.norm(v_masked)
    if a == 0 or b == 0:
        return 0
    return np.dot(u_masked, v_masked)/(a*b)

def euclidean_similarity(u, v):
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) == 0:
        return 0
    d = np.linalg.norm(u[mask] - v[mask])
    return 1 / (1 + d)

def manhattan_similarity(u, v):
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) == 0:
        return 0
    d = np.sum(np.abs(u[mask] - v[mask]))
    return 1 / (1 + d)

def pearson_similarity(u, v):
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) < 2:
        return 0
    u_masked = u[mask]
    v_masked = v[mask]
    u_mean = np.mean(u_masked)
    v_mean = np.mean(v_masked)
    num = np.dot(u_masked - u_mean, v_masked - v_mean)
    denom = np.linalg.norm(u_masked - u_mean) * np.linalg.norm(v_masked - v_mean)
    if denom == 0:
        return 0
    return num / denom

def adjusted_cosine_similarity(i, j, users_items):
    ratings_i = users_items[:, i]
    ratings_j = users_items[:, j]

    mask = ~np.isnan(ratings_i) & ~np.isnan(ratings_j)
    if np.sum(mask) < 2:
        return 0
    
    user_means = np.nanmean(users_items, axis=1)
    ratings_i_adj = ratings_i[mask] - user_means[mask]
    ratings_j_adj = ratings_j[mask] - user_means[mask]

    num = np.dot(ratings_i_adj, ratings_j_adj)
    denom = np.linalg.norm(ratings_i_adj) * np.linalg.norm(ratings_i_adj)

    if denom == 0:
        return 0
    return num / denom

def jaccard_similarity(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    intersect = np.sum((a==1) & (b==1))
    union = np.sum((a==1) | (b==1))
    if union == 0:
        return 0
    return intersect / union

def log_likelihood_similarity(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]

    k11 = np.sum((a==1) & (b==1))
    k10 = np.sum((a==1) & (b==0))
    k01 = np.sum((a==0) & (b==1))
    k00 = np.sum((a==0) & (b==0))
    N = k11 + k10 + k01 + k00

    row1 = k11 + k10
    row2 = k01 + k00
    col1 = k11 + k01
    col2 = k10 + k00
    E11 = row1 * col1 / N if N else 0
    E10 = row1 * col2 / N if N else 0
    E01 = row2 * col1 / N if N else 0
    E00 = row2 * col2 / N if N else 0

    def term(k, e):
        return 0 if k == 0 else k * np.log(k / e)
    
    llr = 2 * (term(k11, E11) + term(k10, E10) + term(k01, E01) + term(k00, E00))
    return llr

def tanimoto_similarity(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    num = np.dot(a, b)
    denom = np.dot(a, a) + np.dot(b, b) - num
    if denom == 0:
        return 0.0
    return num / denom

def hamming_similarity(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return 0.0
    return np.mean(a==b)