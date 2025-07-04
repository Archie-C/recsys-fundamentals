import time


t0 = time.perf_counter()

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.als import train_simple_explicit_biased_als
from src.utils.data_loading import load_split
from src.hybrid import evaluate_hybrid


R_train, R_test, n_users, n_items, train, test, global_mean = load_split(
    "data/raw/ml-100k/u1.base", "data/raw/ml-100k/u1.test", mean_centered=True
)

item_cols = [f'genre_{i}' for i in range(19)]
items = pd.read_csv('data/raw/ml-100k/u.item', sep='|', names=['movie_id','title','release_date','video_release','IMDb_URL'] + item_cols, encoding='latin-1')

k = 30
lam = 0.05
n_iter = 12
mu, bu, bi, X, Y = train_simple_explicit_biased_als(R_train, k=k, lam=lam, lam_bias=lam, n_iter=n_iter, seed=42)

metrics = evaluate_hybrid(R_train, R_test,mu, bu, bi,  X, Y, items, item_cols, alpha= 0.1, k=10)
elapsed = time.perf_counter() - t0
print(f"Computation time: {elapsed:.2f} s")
print(f"Hybrid (k={k}, λ={lam}, iters={n_iter})")
print(metrics)