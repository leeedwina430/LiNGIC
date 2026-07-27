import numpy as np
from functools import partial
from itertools import combinations_with_replacement
from math import comb, factorial, sqrt
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel
import multiprocessing
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Kernel construction and centering
# ---------------------------------------------------------------------------

# Build the explicit first- and second-order feature kernel used by the
# "direct" kernel option.
def psi_kernel_matrix(Y, add_norm2=True):
    Y = np.asarray(Y)
    K1 = Y @ Y.T
    Y2 = Y**2
    K2 = Y2 @ Y2.T
    if add_norm2:
        r2 = np.sum(Y**2, axis=1, keepdims=True)  # Shape: (n, 1).
        K3 = r2 @ r2.T
        return K1 + K2 + K3
    else:
        return K1 + K2


# Double-center a square Gram matrix.
def center_kernel_matrix(K: np.ndarray):
    n = np.shape(K)[0]
    K_colsums = K.sum(axis=0)
    K_allsum = K_colsums.sum()
    return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)


# Standardize the observations and construct a causal-learn Gram matrix.
# Constant input dimensions are mapped to zero after standardization.
def causallearn_kernel_matrix(data_x, kernel='poly', degree=2):
    if kernel == 'poly':
        kernelX = PolynomialKernel(degree=degree, const=1)
    elif kernel == 'gauss':
        kernelX = GaussianKernel()
        kernelX.set_width_empirical_hsic(data_x)
    elif kernel == 'direct':
        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.  # Handle constant input dimensions.
        return psi_kernel_matrix(data_x, add_norm2=True)
    data_x = stats.zscore(data_x, ddof=1, axis=0)
    data_x[np.isnan(data_x)] = 0.  # Handle constant input dimensions.
    Kx = kernelX.kernel(data_x)
    return Kx


# ---------------------------------------------------------------------------
# One-way Gaussian-polynomial statistic
# ---------------------------------------------------------------------------

# Compute HSIC with the first kernel on X and the second kernel on Y.
def gauss_poly_HSIC(data_x, data_y, kernel=['gauss', 'poly'], degree=[-1, 2]):
    Kx = causallearn_kernel_matrix(data_x, kernel=kernel[0], degree=degree[0])
    Ky = causallearn_kernel_matrix(data_y, kernel=kernel[1], degree=degree[1])
    K_tilde = center_kernel_matrix(Kx)
    L_tilde = center_kernel_matrix(Ky)
    return np.sum(K_tilde * L_tilde) / (Kx.shape[0] ** 2)


# Estimate the one-way permutation p-value using multiprocessing.
def gauss_poly_HSIC_pvalue_parallel(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm_parallel(
        partial(gauss_poly_HSIC, kernel=kernel),
        gauss_poly_HSIC(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )
    return p


# Estimate the one-way permutation p-value sequentially.
def gauss_poly_HSIC_pvalue(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm(
        partial(gauss_poly_HSIC, kernel=kernel),
        gauss_poly_HSIC(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )
    return p


# ---------------------------------------------------------------------------
# Two-way Gaussian-polynomial statistic
# ---------------------------------------------------------------------------

# Add the Gaussian(X)-polynomial(Y) and Gaussian(Y)-polynomial(X) statistics.
def gauss_poly_HSIC_twice(data_x, data_y, kernel=['gauss', 'poly'], degree=[-1, 2]):
    Kx = causallearn_kernel_matrix(data_x, kernel=kernel[0], degree=degree[0])
    Ky = causallearn_kernel_matrix(data_y, kernel=kernel[1], degree=degree[1])
    K_tilde = center_kernel_matrix(Kx)
    L_tilde = center_kernel_matrix(Ky)
    GxPy_hsic = np.sum(K_tilde * L_tilde) / (Kx.shape[0] ** 2)

    Ky = causallearn_kernel_matrix(data_y, kernel=kernel[0], degree=degree[0])
    Kx = causallearn_kernel_matrix(data_x, kernel=kernel[1], degree=degree[1])
    K_tilde = center_kernel_matrix(Kx)
    L_tilde = center_kernel_matrix(Ky)
    GyPx_hsic = np.sum(K_tilde * L_tilde) / (Kx.shape[0] ** 2)
    return GxPy_hsic + GyPx_hsic


# Estimate the two-way permutation p-value using multiprocessing.
def gauss_poly_HSIC_twice_pvalue_parallel(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm_parallel(
        partial(gauss_poly_HSIC_twice, kernel=kernel),
        gauss_poly_HSIC_twice(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )
    return p


# Estimate the two-way permutation p-value sequentially.
def gauss_poly_HSIC_twice_pvalue(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm(
        partial(gauss_poly_HSIC_twice, kernel=kernel),
        gauss_poly_HSIC_twice(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )
    return p


# ---------------------------------------------------------------------------
# Sample-symmetrized Gaussian-polynomial statistic
# ---------------------------------------------------------------------------

# Concatenate X with Y and Y with X before applying the asymmetric kernels.
def gauss_poly_HSIC_sample_sym(data_x, data_y, kernel=['gauss', 'poly'], degree=[-1, 2]):
    data_xy = np.hstack([data_x, data_y])
    data_yx = np.hstack([data_y, data_x])

    Kx = causallearn_kernel_matrix(data_xy, kernel=kernel[0], degree=degree[0])
    Ky = causallearn_kernel_matrix(data_yx, kernel=kernel[1], degree=degree[1])
    K_tilde = center_kernel_matrix(Kx)
    L_tilde = center_kernel_matrix(Ky)
    return np.sum(K_tilde * L_tilde) / (Kx.shape[0] ** 2)


# Estimate the sample-symmetrized permutation p-value using multiprocessing.
def gauss_poly_HSIC_sample_sym_pvalue_parallel(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm_parallel(
        partial(gauss_poly_HSIC_sample_sym, kernel=kernel),
        gauss_poly_HSIC_sample_sym(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )

    return p


# Estimate the sample-symmetrized permutation p-value sequentially.
def gauss_poly_HSIC_sample_sym_pvalue(data_x, data_y, kernel=['gauss', 'poly'], N_PERM=500):
    p = compute_pvalue_perm(
        partial(gauss_poly_HSIC_sample_sym, kernel=kernel),
        gauss_poly_HSIC_sample_sym(data_x, data_y, kernel=kernel),
        data_x,
        data_y,
        n_perm=N_PERM,
    )
    return p


# ---------------------------------------------------------------------------
# Permutation calibration
# ---------------------------------------------------------------------------

# Sequential permutation calibration. The strict ">" comparison and the
# original p-value formula are intentionally preserved.
def compute_pvalue_perm(method, test_stat, x, y, n_perm=2000, seed=42):
    stats = []
    temp_y = y.copy()
    for i in range(n_perm):
        perm_index = np.random.permutation(list(range(y.shape[0])))
        temp_y = y[perm_index]
        temp_stat = method(x, temp_y)
        stats.append(temp_stat)

    p = np.sum(np.array(stats) > test_stat) / n_perm
    return p


# Evaluate one permutation from an explicit random seed.
def _one_perm(args):
    method, x, y, seed = args
    rng = np.random.default_rng(seed)
    perm_index = rng.permutation(y.shape[0])
    temp_y = y[perm_index]
    return method(x, temp_y)


# Parallel permutation calibration with deterministic per-permutation seeds.
def compute_pvalue_perm_parallel(method, test_stat, x, y, n_perm=2000, n_jobs=16, seed=42):
    args_list = [
        (method, x, y, seed + i) for i in range(n_perm)
    ]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        stats = list(pool.imap(_one_perm, args_list))
    stats = np.array(stats)
    p = np.sum(stats > test_stat) / n_perm
    return p


