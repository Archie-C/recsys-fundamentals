import time
t0 = time.perf_counter()

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.als import train_simple_explicit_als
from src.utils.data_loading import load_split
from src.metrics.evaluate import evaluate_XY

R_train, R_test, n_users, n_items, train, test, global_mean = load_split("data/raw/ml-100k/u1.base", "data/raw/ml-100k/u1.test", mean_centered=True)

k = 30
lam = 0.05
n_iter = 12
X, Y = train_simple_explicit_als(R_train, k=k, lam=lam, n_iter=n_iter, seed=42)

predictions, actuals = [], []
skipped = 0

for row in test.itertuples():
    u = row.user_id - 1
    i = row.item_id - 1
    if u >= X.shape[0] or i >= Y.shape[0]:
        skipped += 1
        predictions.append(np.nan)
    else:
        pred = np.clip(global_mean + X[u] @ Y[i], 1, 5)
        predictions.append(pred)
    actuals.append(row.rating)

pred = np.array(predictions)
act  = np.array(actuals)
mask = ~np.isnan(pred)

rmse = np.sqrt(mean_squared_error(act[mask], pred[mask]))
mae  = mean_absolute_error(act[mask], pred[mask])
metrics = evaluate_XY(R_train, R_test, X, Y, k=10)

elapsed = time.perf_counter() - t0

# With default settings
# Computation time: 1.77 s
# Evaluated 20000 of 20000 test cases (skipped 0)
# Explicit ALS (k=30, λ=0.05, iters=12)
# RMSE: 1.7610
# MAE : 1.2550

print(f"Computation time: {elapsed:.2f} s")
print(f"Evaluated {mask.sum()} of {len(pred)} test cases (skipped {skipped})")
print(f"Explicit ALS (k={k}, λ={lam}, iters={n_iter})")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(metrics)