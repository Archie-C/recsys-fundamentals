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

def train_simple_explicit_biased_als(R, k=20, lam=0.1, lam_bias=0.01, n_iter=10, seed=0, init_X=None, init_Y=None, init_bu=None, init_bi=None):
    m, n = R.shape
    rng = np.random.default_rng(seed)

    mu = R.data.mean()
    X = init_X if init_X is not None else 0.01 * rng.standard_normal((m, k))
    Y = init_Y if init_Y is not None else 0.01 * rng.standard_normal((n, k))
    bu = init_bu if init_bu is not None else np.zeros(m)
    bi = init_bi if init_bi is not None else np.zeros(n)

    Rt = R.T.tocsr()
    I_k = np.eye(k)

    for _ in range(n_iter):

        # ---------- (1) update user-biases ----------
        for u in range(m):
            start, end = R.indptr[u], R.indptr[u+1]
            if start == end:
                continue
            idx   = R.indices[start:end]                # items rated by u
            r_u   = R.data[start:end]
            resid = r_u - mu - bi[idx] - X[u] @ Y[idx].T
            bu[u] = resid.sum() / (len(idx) + lam_bias)

        # ---------- (2) update item-biases ----------
        for i in range(n):
            start, end = Rt.indptr[i], Rt.indptr[i+1]
            if start == end:
                continue
            idx   = Rt.indices[start:end]               # users who rated i
            r_i   = Rt.data[start:end]
            resid = r_i - mu - bu[idx] - X[idx] @ Y[i]
            bi[i] = resid.sum() / (len(idx) + lam_bias)

        # ---------- (3) update user factors ----------
        for u in range(m):
            start, end = R.indptr[u], R.indptr[u+1]
            if start == end:
                continue
            idx   = R.indices[start:end]
            Y_u   = Y[idx]                              # |I_u| × k
            r_hat = R.data[start:end] - mu - bu[u] - bi[idx]
            A = Y_u.T @ Y_u + lam * I_k
            b = Y_u.T @ r_hat
            X[u] = np.linalg.solve(A, b)

        # ---------- (4) update item factors ----------
        for i in range(n):
            start, end = Rt.indptr[i], Rt.indptr[i+1]
            if start == end:
                continue
            idx   = Rt.indices[start:end]
            X_i   = X[idx]                              # |J_i| × k
            r_hat = Rt.data[start:end] - mu - bu[idx] - bi[i]
            A = X_i.T @ X_i + lam * I_k
            b = X_i.T @ r_hat
            Y[i] = np.linalg.solve(A, b)

    return mu, bu, bi, X, Y