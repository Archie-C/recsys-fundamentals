import pandas as pd
from scipy import sparse
import numpy as np

def df_to_csr(df, n_users, n_items, mean_centered=False):
    rows = df.user_id.values - 1
    cols = df.item_id.values - 1
    # cast to float so we can subtract the mean
    data = df.rating.values.astype(float)
    if mean_centered:
        global_mean = data.mean()
        data -= global_mean
        return sparse.coo_matrix((data, (rows, cols)),
                                 shape=(n_users, n_items)).tocsr(), global_mean
    else:
        return sparse.coo_matrix((data, (rows, cols)),
                                 shape=(n_users, n_items)).tocsr()


def load_split(base_path, test_path, mean_centered=False):
    names = ["user_id", "item_id", "rating", "timestamp"]
    train = pd.read_csv(base_path, sep="\t", names=names)
    test = pd.read_csv(test_path, sep="\t", names=names)

    n_users = max(train.user_id.max(), test.user_id.max())
    n_items = max(train.item_id.max(), test.item_id.max())

    if mean_centered:
        R_train, global_mean = df_to_csr(train, n_users, n_items, mean_centered=mean_centered)
        R_test, _ = df_to_csr(test, n_users, n_items, mean_centered=mean_centered)
    else:
        R_train = df_to_csr(train, n_users, n_items, mean_centered=mean_centered)
        R_test = df_to_csr(test, n_users, n_items, mean_centered=mean_centered)
        global_mean = None

    return R_train, R_test, n_users, n_items, train, test, global_mean

    
