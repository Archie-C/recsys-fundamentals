import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from src.content_based_baseline import compute_user_profiles

def test_compute_user_profiles_simple():
    item_cols = ['f1', 'f2']
    items = pd.DataFrame({
        'f1': [1, 3, 5],
        'f2': [2, 4, 6]
    })

    # Construct R: 2 users × 3 items
    #   User 0 ratings: [4, 1, 5] → likes items 0 & 2
    #   User 1 ratings: [3, 4, 2] → likes item 1 only
    ratings = np.array([[4, 1, 5],
                        [3, 4, 2]])
    R = csr_matrix(ratings)

    # Hand-compute expected profiles:
    #   User 0: avg([1,2], [5,6]) = [3,4]
    #   User 1: [3,4]
    expected = pd.DataFrame({
        'user_id': [0, 1],
        'f1': [3.0, 3.0],
        'f2': [4.0, 4.0]
    })

    actual = compute_user_profiles(R, items, item_cols)

    
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)