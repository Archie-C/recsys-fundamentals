import numpy as np
import scipy.sparse as sp
import pytest

from src.als import train_simple_explicit_als

def mf_loss(R_csr, X, Y, lam):
    R = R_csr.tocoo()
    pred = np.sum(X[R.row] * Y[R.col], axis=1)
    se = np.square(R.data - pred).sum()
    reg = lam * (np.square(X).sum() + np.square(Y).sum())
    return se + reg

@pytest.fixture(scope="module")
def tiny_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(8, 2))
    Y0 = rng.normal(size=(10, 2))
    full = X0 @ Y0.T
    rows, cols = np.where(np.ones_like(full))
    R = sp.coo_matrix((full[rows, cols], (rows, cols))).tocsr()
    return R, X0, Y0

def test_shapes(tiny_data):
    R, *_ = tiny_data
    X, Y = train_simple_explicit_als(R, k=2, lam=0.1, n_iter=1)
    m, n = R.shape
    assert X.shape == (m, 2)
    assert Y.shape == (n, 2)

def test_loss_decreases(tiny_data):
    R, *_ = tiny_data
    m, n  = R.shape
    X, Y  = np.random.randn(m, 2)*0.01, np.random.randn(n, 2)*0.01
    lam   = 0.1
    prev  = mf_loss(R, X, Y, lam)

    # run 4 outer iterations one by one
    for _ in range(4):
        X, Y = train_simple_explicit_als(R, k=2, lam=lam, n_iter=1, init_X=X, init_Y=Y)
        curr = mf_loss(R, X, Y, lam)
        assert curr < prev - 1e-8, "loss did not drop"
        prev = curr

def test_synthetic_rmse(tiny_data):
    R, X0, Y0 = tiny_data
    X, Y = train_simple_explicit_als(R, k=2, lam=1e-4, n_iter=15, seed=1)
    full_pred = X @ Y.T
    rmse = np.sqrt(((full_pred - (X0 @ Y0.T))**2).mean())
    assert rmse < 1e-2