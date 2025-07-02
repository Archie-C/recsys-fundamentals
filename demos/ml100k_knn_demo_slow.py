import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.similarities import (
    cosine_similarity, pearson_similarity, euclidean_similarity, manhattan_similarity
)

from src.knn import knn_predict_item, knn_predict_user

train = pd.read_csv("data/raw/ml-100k/u1.base", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
test = pd.read_csv("data/raw/ml-100k/u1.test", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

n_users = max(train.user_id.max(), test.user_id.max())
n_items = max(train.item_id.max(), test.item_id.max())

user_item = np.full((n_users, n_items), np.nan)
for row in train.itertuples():
    user_item[row.user_id - 1, row.item_id - 1] = row.rating

k = 10
similarity_func = cosine_similarity

predictions = []
actuals = []
skipped = 0

for row in test.itertuples():
    user_idx = row.user_id - 1
    item_idx = row.item_id - 1
    pred = knn_predict_user(user_item, user_idx, item_idx, k=k, similarity_func=similarity_func)
    predictions.append(pred)
    actuals.append(row.rating)
    if np.isnan(pred):
        skipped += 1

pred = np.array(predictions)
act = np.array(actuals)
mask = ~np.isnan(pred)
rmse = np.sqrt(mean_squared_error(act[mask], pred[mask]))
mae = mean_absolute_error(act[mask], pred[mask])

print(f"Evaluated {mask.sum()} of {len(pred)} test cases (skipped {skipped} due to cold-starts)")
print(f"User-based kNN ({similarity_func.__name__}, k={k})")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")