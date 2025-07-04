[![CI](https://github.com/Archie-C/recsys-fundamentals/actions/workflows/python-test.yaml/badge.svg)](https://github.com/Archie-C/recsys-fundamentals/actions/workflows/python-test.yaml)

# Recsys Sprint

**A 14-day hands-on sprint for building and evaluating recommender-system baselines.**

## 📂 Project Structure
```
├── Makefile
├── README.md
├── requirements.txt
├── get_data.py
├── src/
│ ├── knn.py
│ ├── als.py
│ └── …
├── tests/
│ ├── test_data.py
│ ├── test_knn.py
│ └── …
├── data/
│ └── raw/ ← ml-100k.zip & extracted files
└── venv/ ← your virtual-environment
```

## ⚙️ Setup
1. Clone the Repo

    ```bash
    git clone <https://github.com/Archie-C/recsys-fundamentals>
    cd recsys-fundamentals
    ```
2. Create & active venv
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

# 🗄️ Download & verify Data
```bash
make data
```
This runs `get_data.py`, which:
1. Downloads `ml-100k.zip`
2. Verifies its MD5 checksum
3. Unpacks into `data/raw/`

# ✅ Testing
```bash
make test
```
Runs `pytest` (via your venv) and ensures data sanity and code correctness

# Results

### CF vs Hybrid
Below are the ranking metrics comparing pure ALS vs. the hybrid CF+genre model (with optimal α).

| Model              | α    | HR@10   | NDCG@10  | Item Coverage |
|--------------------|------|---------|----------|---------------|
| Pure ALS           | —    | 0.0976  | 0.0471   | 0.289         |
| Hybrid (genres)    | 0.1 | 0.2333  | 0.1272   | 0.408         |

* α = weight on CF score (1 = pure CF, 0 = pure content).  
* Metrics computed on the ML-100K test split, k=10 recommendations.

**Runtime:** CF only: ~2.4 s | Hybrid: ~3.0 s

# Fundamental Mathematics

[Singular Value Decomposition](https://cookie-aura-4c6.notion.site/Singular-Value-Decomposition-in-Recommender-Systems-223acccb70f1808d8724c6f74cc6b7b1)

[Basic kNN](https://cookie-aura-4c6.notion.site/K-Nearest-Neighbours-in-Recommender-Systems-224acccb70f1802badbec27c131b0689?pvs=73)

[Matrix Factorising and ALS](https://cookie-aura-4c6.notion.site/Matrix-Factorising-and-Alternating-Least-Squares-ALS-224acccb70f18040b7c1c5312128a01f?pvs=73)

[Statistical Tests](https://cookie-aura-4c6.notion.site/Statistical-Tests-in-Recommender-Systems-225acccb70f180b693b3d0716e3162a0?pvs=73)

[Linear and Logistic Regression](https://cookie-aura-4c6.notion.site/Linear-and-Logistic-Regression-226acccb70f180a19ca4eeede46ac5ca?pvs=73)