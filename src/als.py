import numpy as np

def train_simple_explicit_als(R, k=20, lam=0.1, n_iter=10, seed=0, init_X=None, init_Y=None):
    m, n = R.shape
    rng = np.random.default_rng(seed)
    X = init_X if init_X is not None else 0.01 * rng.standard_normal((m, k))
    Y = init_Y if init_Y is not None else 0.01 * rng.standard_normal((n, k))

    Rt = R.T.tocsr()

    I_k = np.eye(k)

    for _ in range(n_iter):
        # user update
        for u in range(m):
            idx = R.indices[R.indptr[u]:R.indptr[u+1]]
            if idx.size == 0:
                continue
            Y_u = Y[idx]
            A = Y_u.T @ Y_u + lam * I_k
            b = Y_u.T @ R.data[R.indptr[u]:R.indptr[u+1]]
            X[u] = np.linalg.solve(A, b)

        # item update
        for i in range(n):
            idx = Rt.indices[Rt.indptr[i]:Rt.indptr[i+1]]
            if idx.size == 0:
                continue
            X_i = X[idx]
            A = X_i.T @ X_i + lam * I_k
            b = X_i.T @ Rt.data[Rt.indptr[i]:Rt.indptr[i+1]]
            Y[i] = np.linalg.solve(A, b)

    return X, Y