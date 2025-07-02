import numpy as np

from src.utils.similarities import cosine_similarity

def knn_predict_user(
    ratings_matrix, user_id, item_id, k=5, similarity_func=cosine_similarity
):
    target_ratings = ratings_matrix[user_id]  # shape (n_items,)
    sims = []
    for other_user in range(ratings_matrix.shape[0]):
        if other_user == user_id:
            continue
        other_ratings = ratings_matrix[other_user]  # shape (n_items,)
        if np.isnan(other_ratings[item_id]):
            continue
        # Both vectors are (n_items,)
        sim = similarity_func(target_ratings, other_ratings)
        sims.append((sim, other_ratings[item_id]))
    sims.sort(reverse=True, key=lambda x: x[0])
    top_k = [x for x in sims[:k] if x[0] > 0]
    if not top_k:
        return np.nan
    sims_arr, ratings_arr = zip(*top_k)
    prediction = np.average(ratings_arr, weights=np.abs(sims_arr))
    return prediction

def knn_predict_item(
    ratings_matrix, user_id, item_id, k=5, similarity_func=cosine_similarity
):
    target_ratings = ratings_matrix[:, item_id]  # shape (n_users,)
    sims = []
    for other_item in range(ratings_matrix.shape[1]):
        if other_item == item_id:
            continue
        if np.isnan(ratings_matrix[user_id, other_item]):
            continue
        other_ratings = ratings_matrix[:, other_item]  # shape (n_users,)
        sim = similarity_func(target_ratings, other_ratings)
        sims.append((sim, ratings_matrix[user_id, other_item]))
    sims.sort(reverse=True, key=lambda x: x[0])
    top_k = [x for x in sims[:k] if x[0] > 0]
    if not top_k:
        return np.nan
    sims_arr, ratings_arr = zip(*top_k)
    prediction = np.average(ratings_arr, weights=np.abs(sims_arr))
    return prediction