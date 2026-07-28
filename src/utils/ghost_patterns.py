"""Hadamard pattern generation utilities"""

import scipy.linalg
import numpy as np
import functools
import logging


@functools.lru_cache(maxsize=32)
def get_hadamard_matrix_cached(N, M, perm_seed=None):
    """
    Generate the Hadamard measurement matrix (cached; perm_seed is part of the cache key)

    Args:
        N: total number of image pixels
        M: number of measurements (bucket_size)
        perm_seed: for the acquisition-order ablation -- when not None, applies a fixed seeded permutation to the M rows

    Returns:
        (M, N) numpy array
    """
    return get_hadamard_matrix(N, M, perm_seed)


def get_hadamard_matrix(N, M, perm_seed=None):
    """
    Generate the Hadamard measurement matrix

    Args:
        N: total number of image pixels
        M: number of measurements (bucket_size)
        perm_seed: when not None, uses an independent RandomState(perm_seed) to apply one
            fixed row permutation to the first M rows (acquisition-order ablation). This function is
            the single source of truth for Φ -- CPU/GPU, training/evaluation all go through here,
            so the permutation stays consistent along the whole chain automatically.

    Returns:
        (M, N) numpy array, the first M Hadamard bases (optionally reordered by perm_seed)
    """
    # find the smallest power of 2
    n = 2 ** int(np.ceil(np.log2(max(N, M))))

    # generate the full Hadamard matrix
    H = scipy.linalg.hadamard(n, dtype=np.float32)

    # normalization (optional, as needed)
    # H = H / np.sqrt(n)

    # take the first M rows and the first N columns
    H_sub = H[:M, :N]

    # acquisition-order ablation: fixed seeded row permutation (independent RNG, leaves the global random state untouched)
    if perm_seed is not None:
        perm = np.random.RandomState(int(perm_seed)).permutation(M)
        H_sub = H_sub[perm]
        logging.info(f"Hadamard row permutation enabled: perm_seed={perm_seed} (M={M})")

    return H_sub.astype(np.float32)