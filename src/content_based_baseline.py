import numpy as np
import pandas as pd
from src.utils.similarities import cosine_similarity

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