# -*- coding: utf-8 -*-
"""ADMM-L1 reconstruction: known-answer test against an INDEPENDENT solver.

`admm_l1_recon` (src/reconstruction/cs.py) is the paper's off-shelf CS reconstruction
baseline (recon_dump.py --method cs; l1=0.01, rho=1.0, steps=100). It previously had
zero test coverage. This pins it to sklearn's coordinate-descent Lasso solving the
IDENTICAL convex program

    min_alpha  0.5 ||A_Psi alpha - y||_2^2 + lam ||alpha||_1

sklearn's `Lasso(alpha=lam/M, fit_intercept=False)` minimiser is the same point (its
objective is (1/2M)||.||^2 + alpha_sk||.||_1, so alpha_sk = lam/M matches). On a
k-sparse-in-DCT signal the two independent solvers must land on the same reconstruction.

Teeth (measured 2026-07-18): correct admm matches the reference at corr = 1.0000;
replacing the z-update's soft-threshold with the identity (dropping the L1 prox) drops
the agreement to 0.946 < 0.99. A wrong Woodbury x-update, a lam*rho (vs lam/rho)
z-threshold, or a broken dual update all move the solution off the reference.
"""
import warnings

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from src.reconstruction.cs import admm_l1_recon, dct_2d_matrix
from src.utils.ghost_patterns import get_hadamard_matrix


def _corr(a, b):
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def test_admm_l1_matches_independent_lasso():
    n, M, lam = 16, 128, 0.01
    N = n * n
    Psi = dct_2d_matrix(n)
    rng = np.random.RandomState(0)
    alpha = np.zeros(N)
    alpha[rng.choice(N, 8, replace=False)] = rng.randn(8)          # 8-sparse in the DCT domain
    x_true = (Psi @ alpha).reshape(n, n).astype(np.float32)
    phi = get_hadamard_matrix(N, M, None)                          # (M, N) natural-order forward
    y = (phi @ x_true.reshape(-1)).astype(np.float32)[None, :]     # (1, M) noiseless measurement

    # reference: the SAME convex program solved by an independent method (coordinate descent)
    A_Psi = phi.astype(np.float32) @ Psi.astype(np.float32)        # (M, N)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)       # CD's last-digit gap is irrelevant at corr>0.99
        ref = Lasso(alpha=lam / M, fit_intercept=False, max_iter=200000, tol=1e-9).fit(A_Psi, y[0])
    img_ref = (ref.coef_ @ Psi.T).reshape(n, n)

    rec = admm_l1_recon(phi, y, n, l1_weight=lam, rho=1.0, steps=400, device='cpu')[0, 0]

    assert np.isfinite(rec).all(), "admm returned non-finite values"
    assert rec.shape == (n, n)
    assert rec.min() >= 0.0 and rec.max() <= 1.0 + 1e-5, "output is min-max normalised to [0,1]"
    assert _corr(rec, img_ref) > 0.99, \
        f"admm disagrees with the independent Lasso reference (corr={_corr(rec, img_ref):.4f})"
