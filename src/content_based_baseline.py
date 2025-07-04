import numpy as np
import pandas as pd
from src.utils.similarities import cosine_similarity
from scipy.sparse import csr_matrix

def compute_user_profiles(R, items, item_cols):
    B = (R >= 4).astype(int)
    counts = np.array(B.sum(axis=1)).flatten()
    counts[counts == 0] = 1

    X = items[item_cols].values

    genre_sums = B.dot(X) 
    
    P = genre_sums / counts[:, None]
    
    n_users = R.shape[0]
    user_profiles = pd.DataFrame(P, columns=item_cols, index=np.arange(n_users))
    user_profiles.index.name = 'user_id'
    user_profiles.reset_index(inplace=True)
    
    return user_profiles

def get_genre_matrix(item_meta_df: pd.DataFrame, genre_cols:list) -> csr_matrix:
    """
    Build a sparse binary item-feature matrix from one-hot genre columns.

    Parameters
    ----------
    item_meta_df : DataFrame with `item_id` (0-based) and binary genre_cols
    genre_cols    : list of column names representing genres (0/1)

    Returns
    -------
    genre_matrix : csr_matrix of shape (n_items, n_genres)
    """
    df = item_meta_df.sort_values('movie_id')
    mat = df[genre_cols].values.astype(float)
    return csr_matrix(mat)

def compute_content_scores(R: csr_matrix, genre_matrix: csr_matrix) -> np.ndarray:
    """
    Build user profiles by averaging binary genre vectors for items each user has rated,
    then compute content similarity scores: user_profiles @ genre_matrix.T

    Returns
    -------
    content_scores : dense array of shape (n_users, n_items)
    """
    ratings = R.toarray()    # shape (n_users, n_items)
    weights = ratings / ratings.sum(axis=1, keepdims=True)  # normalize ratings
    weights[np.isnan(weights)] = 0
    user_profiles = weights @ genre_matrix
    content_scores = user_profiles @ genre_matrix.T
    return content_scores