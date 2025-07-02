import numpy as np
from src.utils.similarities import cosine_similarity, pearson_similarity
from src.knn import knn_predict_user, knn_predict_item

ratings_matrix = np.array([
    [5, 3, np.nan, 1],
    [4, np.nan, np.nan, 1],
    [1, 1, np.nan, 5],
    [1, np.nan, np.nan, 4],
    [np.nan, 1, 5, 4]
])

def test_knn_user_cosine():
    # Predict user 0's rating for item 2 using cosine similarity, k=2
    pred = knn_predict_user(ratings_matrix, user_id=0, item_id=2, k=2, similarity_func=cosine_similarity)
    # Only user 4 has rated item 2 (rating=5), check if that's the prediction
    assert np.isclose(pred, 5.0)

def test_knn_item_cosine():
    # Predict user 0's rating for item 2 using item-based kNN, cosine, k=2
    pred = knn_predict_item(ratings_matrix, user_id=0, item_id=2, k=2, similarity_func=cosine_similarity)
    # Only item 1 and item 2 are both rated by user 0, item 2 is nan, so fallback expected
    assert np.isnan(pred) or 1 <= pred <= 5

def test_knn_user_pearson():
    pred = knn_predict_user(ratings_matrix, user_id=0, item_id=2, k=2, similarity_func=pearson_similarity)
    assert np.isnan(pred)

def test_knn_item_pearson():
    pred = knn_predict_item(ratings_matrix, user_id=4, item_id=1, k=2, similarity_func=pearson_similarity)
    assert np.isnan(pred)

def test_knn_user_no_neighbors():
    # User 2, item 2: nobody else rated item 2 except user 4, who is the only neighbor
    # If user 4's rating is NaN, should return NaN
    tmp = ratings_matrix.copy()
    tmp[4, 2] = np.nan
    pred = knn_predict_user(tmp, user_id=2, item_id=2, k=2, similarity_func=cosine_similarity)
    assert np.isnan(pred)