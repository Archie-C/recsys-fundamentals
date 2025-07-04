import time
t0 = time.perf_counter()

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.data_loading import load_split
from src.metrics.evaluate import evaluate_XY, evaluate_content_all_metrics
from src.content_based_baseline import compute_user_profiles

R_train, R_test, n_users, n_items, train, test, global_mean = load_split(
    "data/raw/ml-100k/u1.base", "data/raw/ml-100k/u1.test", mean_centered=True
)

item_cols = [f'genre_{i}' for i in range(19)]
items = pd.read_csv('data/raw/ml-100k/u.item', sep='|', names=['movie_id','title','release_date','video_release','IMDb_URL'] + item_cols, encoding='latin-1')

user_profiles_df = compute_user_profiles(R_train, items, item_cols)

metrics = evaluate_content_all_metrics(user_profiles_df, R_test, items, item_cols, k=10, rating_min=1, rating_max=5)

print(metrics)