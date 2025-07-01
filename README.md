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

# Fundamental Mathematics

[Singular Value Decomposition](https://cookie-aura-4c6.notion.site/Singular-Value-Decomposition-in-Recommender-Systems-223acccb70f1808d8724c6f74cc6b7b1)