"""Numba-accelerated CPU kernels for density simulation and the correlation gradient.

These are the heavy numerical routines behind :class:`openfit.DensityMap`. They are
private (import via the engine, not directly). The GPU (``device="cuda"``) variants
land in the performance phase; ``cuda_erf`` below is the first piece, guarded so the
package still imports on hosts without a CUDA toolkit.
"""

import logging
import math

import numpy as np
from numba import float64, jit, prange, vectorize


@vectorize([float64(float64)], nopython=True)
def numba_erf(x):
    return math.erf(x)


# GPU path (future work, see performance roadmap). Building the CUDA ufunc
# requires a CUDA toolkit; guard it so importing openfit still works on hosts
# without a GPU (e.g. CI runners). ``cuda_erf`` is None when unavailable.
try:

    @vectorize([float64(float64)], target="cuda")
    def cuda_erf(x):
        return math.erf(x)

except Exception:  # pragma: no cover - depends on CUDA availability
    cuda_erf = None
    logging.debug("CUDA not available; cuda_erf disabled.", exc_info=True)


@jit(nopython=True)
def substract_and_fold(arr, p):
    darr = arr[:, 1:] - arr[:, :-1]
    darr[:, -2 * p : -p] += darr[:, :p]
    darr[:, p : 2 * p] += darr[:, -p:]
    return darr[:, p:-p]


@jit(nopython=True, parallel=True)
def sim_map(coordinates, n_voxels, voxel_size, sigma, epsilon, padding, multiplier):
    n_dim = coordinates.shape[0]
    i_dim = n_voxels[0]
    j_dim = n_voxels[1]
    k_dim = n_voxels[2]

    voxel_limits_x = np.arange(-padding, n_voxels[0] + 1 + padding) * voxel_size[0]
    voxel_limits_y = np.arange(-padding, n_voxels[1] + 1 + padding) * voxel_size[1]
    voxel_limits_z = np.arange(-padding, n_voxels[2] + 1 + padding) * voxel_size[2]

    min_coords = coordinates - multiplier * sigma
    max_coords = coordinates + multiplier * sigma

    limits = np.zeros((coordinates.shape[0], 6), dtype=np.int64)

    limits[:, 0] = np.searchsorted(voxel_limits_x, min_coords[:, 0]) - 1
    limits[:, 1] = np.searchsorted(voxel_limits_x, max_coords[:, 0]) + 1
    limits[:, 2] = np.searchsorted(voxel_limits_y, min_coords[:, 1]) - 1
    limits[:, 3] = np.searchsorted(voxel_limits_y, max_coords[:, 1]) + 1
    limits[:, 4] = np.searchsorted(voxel_limits_z, min_coords[:, 2]) - 1
    limits[:, 5] = np.searchsorted(voxel_limits_z, max_coords[:, 2]) + 1

    sigma = sigma * np.sqrt(2)  # (3,)
    x_mu_sigma = np.zeros((n_dim, voxel_limits_x.shape[0]))
    y_mu_sigma = np.zeros((n_dim, voxel_limits_y.shape[0]))
    z_mu_sigma = np.zeros((n_dim, voxel_limits_z.shape[0]))
    for n in prange(n_dim):
        x_mu_sigma[n, :] = (voxel_limits_x - coordinates[n, 0]) / sigma[n, 0]  # (n,x+1+2*p)
        y_mu_sigma[n, :] = (voxel_limits_y - coordinates[n, 1]) / sigma[n, 1]  # (n,x+1+2*p)
        z_mu_sigma[n, :] = (voxel_limits_z - coordinates[n, 2]) / sigma[n, 2]  # (n,x+1+2*p)

    phix = (1 + numba_erf(x_mu_sigma)) / 2  # (n,x+1+2*p)
    phiy = (1 + numba_erf(y_mu_sigma)) / 2  # (n,y+1+2*p)
    phiz = (1 + numba_erf(z_mu_sigma)) / 2  # (n,z+1+2*p)

    dphix_dx = np.zeros((n_dim, voxel_limits_x.shape[0]))  # (n,x+1+2*p)
    dphiy_dy = np.zeros((n_dim, voxel_limits_y.shape[0]))  # (n,y+1+2*p)
    dphiz_dz = np.zeros((n_dim, voxel_limits_z.shape[0]))  # (n,z+1+2*p)
    for n in prange(n_dim):
        dphix_dx[n, :] = -np.exp(-x_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 0]  # (n,x+1+2*p)
        dphiy_dy[n, :] = -np.exp(-y_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 1]  # (n,y+1+2*p)
        dphiz_dz[n, :] = -np.exp(-z_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 2]  # (n,z+1+2*p)

    dphix = substract_and_fold(phix, padding)  # (n,x)
    dphiy = substract_and_fold(phiy, padding)  # (n,y)
    dphiz = substract_and_fold(phiz, padding)  # (n,z)

    # Calculate sim
    sim = np.zeros((k_dim, j_dim, i_dim), dtype=np.float64)  # (z,y,x)
    for n in range(n_dim):
        eps_n = epsilon[n]
        i_min, i_max, j_min, j_max, k_min, k_max = limits[n]
        for k in prange(k_min, k_max + 1):
            k = (k - padding) % k_dim
            for j in prange(j_min, j_max + 1):
                j = (j - padding) % j_dim
                for i in prange(i_min, i_max + 1):
                    i = (i - padding) % i_dim
                    sim[k, j, i] += eps_n * dphix[n, i] * dphiy[n, j] * dphiz[n, k]
    return sim


@jit(nopython=True, parallel=True)
def dcorr_v3(coordinates, n_voxels, voxel_size, sigma, epsilon, experimental_map, padding, multiplier):
    n_dim = coordinates.shape[0]
    i_dim = n_voxels[0]
    j_dim = n_voxels[1]
    k_dim = n_voxels[2]

    voxel_limits_x = np.arange(-padding, n_voxels[0] + 1 + padding) * voxel_size[0]
    voxel_limits_y = np.arange(-padding, n_voxels[1] + 1 + padding) * voxel_size[1]
    voxel_limits_z = np.arange(-padding, n_voxels[2] + 1 + padding) * voxel_size[2]

    min_coords = coordinates - multiplier * sigma
    max_coords = coordinates + multiplier * sigma

    limits = np.zeros((coordinates.shape[0], 6), dtype=np.int64)

    limits[:, 0] = np.searchsorted(voxel_limits_x, min_coords[:, 0]) - 1
    limits[:, 1] = np.searchsorted(voxel_limits_x, max_coords[:, 0]) + 1
    limits[:, 2] = np.searchsorted(voxel_limits_y, min_coords[:, 1]) - 1
    limits[:, 3] = np.searchsorted(voxel_limits_y, max_coords[:, 1]) + 1
    limits[:, 4] = np.searchsorted(voxel_limits_z, min_coords[:, 2]) - 1
    limits[:, 5] = np.searchsorted(voxel_limits_z, max_coords[:, 2]) + 1

    sigma = sigma * np.sqrt(2)  # (3,)
    x_mu_sigma = np.zeros((n_dim, voxel_limits_x.shape[0]))
    y_mu_sigma = np.zeros((n_dim, voxel_limits_y.shape[0]))
    z_mu_sigma = np.zeros((n_dim, voxel_limits_z.shape[0]))
    for n in prange(n_dim):
        x_mu_sigma[n, :] = (voxel_limits_x - coordinates[n, 0]) / sigma[n, 0]  # (n,x+1+2*p)
        y_mu_sigma[n, :] = (voxel_limits_y - coordinates[n, 1]) / sigma[n, 1]  # (n,x+1+2*p)
        z_mu_sigma[n, :] = (voxel_limits_z - coordinates[n, 2]) / sigma[n, 2]  # (n,x+1+2*p)

    phix = (1 + numba_erf(x_mu_sigma)) / 2  # (n,x+1+2*p)
    phiy = (1 + numba_erf(y_mu_sigma)) / 2  # (n,y+1+2*p)
    phiz = (1 + numba_erf(z_mu_sigma)) / 2  # (n,z+1+2*p)

    dphix_dx = np.zeros((n_dim, voxel_limits_x.shape[0]))  # (n,x+1+2*p)
    dphiy_dy = np.zeros((n_dim, voxel_limits_y.shape[0]))  # (n,y+1+2*p)
    dphiz_dz = np.zeros((n_dim, voxel_limits_z.shape[0]))  # (n,z+1+2*p)
    for n in prange(n_dim):
        dphix_dx[n, :] = -np.exp(-x_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 0]  # (n,x+1+2*p)
        dphiy_dy[n, :] = -np.exp(-y_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 1]  # (n,y+1+2*p)
        dphiz_dz[n, :] = -np.exp(-z_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 2]  # (n,z+1+2*p)

    dphix_ds = x_mu_sigma * dphix_dx * np.sqrt(2)  # (n,x+1+2*p)
    dphiy_ds = y_mu_sigma * dphiy_dy * np.sqrt(2)  # (n,y+1+2*p)
    dphiz_ds = z_mu_sigma * dphiz_dz * np.sqrt(2)  # (n,z+1+2*p)

    dphix = substract_and_fold(phix, padding)  # (n,x)
    dphiy = substract_and_fold(phiy, padding)  # (n,y)
    dphiz = substract_and_fold(phiz, padding)  # (n,z)

    ddphix_dx = substract_and_fold(dphix_dx, padding)  # (n,x)
    ddphiy_dy = substract_and_fold(dphiy_dy, padding)  # (n,y)
    ddphiz_dz = substract_and_fold(dphiz_dz, padding)  # (n,z)

    ddphix_ds = substract_and_fold(dphix_ds, padding)  # (n,x)
    ddphiy_ds = substract_and_fold(dphiy_ds, padding)  # (n,y)
    ddphiz_ds = substract_and_fold(dphiz_ds, padding)  # (n,z)

    exp = experimental_map  # (z,y,x)

    # Calculate sim
    sim = np.zeros((k_dim, j_dim, i_dim), dtype=np.float64)  # (z,y,x)
    for n in prange(n_dim):
        eps_n = epsilon[n]
        i_min, i_max, j_min, j_max, k_min, k_max = limits[n]
        for k in range(k_min, k_max + 1):
            k = (k - padding) % k_dim
            for j in range(j_min, j_max + 1):
                j = (j - padding) % j_dim
                for i in range(i_min, i_max + 1):
                    i = (i - padding) % i_dim
                    sim[k, j, i] += eps_n * dphix[n, i] * dphiy[n, j] * dphiz[n, k]

    # Normalize sim
    sim_mean = np.mean(sim)
    sim_std = np.std(sim)
    sim = (sim - sim_mean) / sim_std

    # Calculate correlation coefficient
    cc = np.mean(sim * exp)

    # Calculate derivatives
    num1 = np.zeros((n_dim, 7), dtype=np.float64)
    num2 = np.zeros((n_dim, 7), dtype=np.float64)
    for n in prange(n_dim):
        eps_n = epsilon[n]
        i_min, i_max, j_min, j_max, k_min, k_max = limits[n]
        for k in range(k_min, k_max + 1):
            k = (k - padding) % k_dim
            for j in range(j_min, j_max + 1):
                j = (j - padding) % j_dim
                for i in range(i_min, i_max + 1):
                    i = (i - padding) % i_dim
                    exp_val = exp[k, j, i]
                    sim_val = sim[k, j, i]
                    num1[n, 0] += eps_n * ddphix_dx[n, i] * dphiy[n, j] * dphiz[n, k] * exp_val
                    num1[n, 1] += eps_n * dphix[n, i] * ddphiy_dy[n, j] * dphiz[n, k] * exp_val
                    num1[n, 2] += eps_n * dphix[n, i] * dphiy[n, j] * ddphiz_dz[n, k] * exp_val
                    num1[n, 3] += eps_n * ddphix_ds[n, i] * dphiy[n, j] * dphiz[n, k] * exp_val
                    num1[n, 4] += eps_n * dphix[n, i] * ddphiy_ds[n, j] * dphiz[n, k] * exp_val
                    num1[n, 5] += eps_n * dphix[n, i] * dphiy[n, j] * ddphiz_ds[n, k] * exp_val
                    num1[n, 6] += dphix[n, i] * dphiy[n, j] * dphiz[n, k] * exp_val
                    num2[n, 0] += eps_n * ddphix_dx[n, i] * dphiy[n, j] * dphiz[n, k] * sim_val
                    num2[n, 1] += eps_n * dphix[n, i] * ddphiy_dy[n, j] * dphiz[n, k] * sim_val
                    num2[n, 2] += eps_n * dphix[n, i] * dphiy[n, j] * ddphiz_dz[n, k] * sim_val
                    num2[n, 3] += eps_n * ddphix_ds[n, i] * dphiy[n, j] * dphiz[n, k] * sim_val
                    num2[n, 4] += eps_n * dphix[n, i] * ddphiy_ds[n, j] * dphiz[n, k] * sim_val
                    num2[n, 5] += eps_n * dphix[n, i] * dphiy[n, j] * ddphiz_ds[n, k] * sim_val
                    num2[n, 6] += dphix[n, i] * dphiy[n, j] * dphiz[n, k] * sim_val

    num2 *= cc  # (n,7)
    den = sim_std * i_dim * j_dim * k_dim  # (,)

    result = (num1 - num2) / den  # (n,7)
    return result


@jit(nopython=True, parallel=True)
def dcorr_force_cc(coordinates, n_voxels, voxel_size, sigma, epsilon, experimental_map, padding, multiplier):
    """Positional (x, y, z) correlation gradient **and** the correlation coefficient, one pass.

    A slimmed-down :func:`dcorr_v3` for the flexible-fitting hot path: it computes only the 3
    derivatives the bias force needs (dropping the sigma/epsilon columns, ~halving the heavy
    second loop) and returns the ``cc`` it evaluates internally — which :func:`dcorr_v3`
    throws away — so the caller never needs a separate :func:`sim_map` rebuild to log the
    correlation. Matches ``dcorr_v3(...)[:, :3]`` to within the parallel scatter's race floor.

    Returns
    -------
    (numpy.ndarray, float)
        The ``(n, 3)`` gradient of the correlation w.r.t. ``(x, y, z)`` and the correlation
        coefficient ``cc``.
    """
    n_dim = coordinates.shape[0]
    i_dim = n_voxels[0]
    j_dim = n_voxels[1]
    k_dim = n_voxels[2]

    voxel_limits_x = np.arange(-padding, n_voxels[0] + 1 + padding) * voxel_size[0]
    voxel_limits_y = np.arange(-padding, n_voxels[1] + 1 + padding) * voxel_size[1]
    voxel_limits_z = np.arange(-padding, n_voxels[2] + 1 + padding) * voxel_size[2]

    min_coords = coordinates - multiplier * sigma
    max_coords = coordinates + multiplier * sigma

    limits = np.zeros((coordinates.shape[0], 6), dtype=np.int64)
    limits[:, 0] = np.searchsorted(voxel_limits_x, min_coords[:, 0]) - 1
    limits[:, 1] = np.searchsorted(voxel_limits_x, max_coords[:, 0]) + 1
    limits[:, 2] = np.searchsorted(voxel_limits_y, min_coords[:, 1]) - 1
    limits[:, 3] = np.searchsorted(voxel_limits_y, max_coords[:, 1]) + 1
    limits[:, 4] = np.searchsorted(voxel_limits_z, min_coords[:, 2]) - 1
    limits[:, 5] = np.searchsorted(voxel_limits_z, max_coords[:, 2]) + 1

    sigma = sigma * np.sqrt(2)
    x_mu_sigma = np.zeros((n_dim, voxel_limits_x.shape[0]))
    y_mu_sigma = np.zeros((n_dim, voxel_limits_y.shape[0]))
    z_mu_sigma = np.zeros((n_dim, voxel_limits_z.shape[0]))
    for n in prange(n_dim):
        x_mu_sigma[n, :] = (voxel_limits_x - coordinates[n, 0]) / sigma[n, 0]
        y_mu_sigma[n, :] = (voxel_limits_y - coordinates[n, 1]) / sigma[n, 1]
        z_mu_sigma[n, :] = (voxel_limits_z - coordinates[n, 2]) / sigma[n, 2]

    phix = (1 + numba_erf(x_mu_sigma)) / 2
    phiy = (1 + numba_erf(y_mu_sigma)) / 2
    phiz = (1 + numba_erf(z_mu_sigma)) / 2

    dphix_dx = np.zeros((n_dim, voxel_limits_x.shape[0]))
    dphiy_dy = np.zeros((n_dim, voxel_limits_y.shape[0]))
    dphiz_dz = np.zeros((n_dim, voxel_limits_z.shape[0]))
    for n in prange(n_dim):
        dphix_dx[n, :] = -np.exp(-x_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 0]
        dphiy_dy[n, :] = -np.exp(-y_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 1]
        dphiz_dz[n, :] = -np.exp(-z_mu_sigma[n, :] ** 2) / np.sqrt(np.pi) / sigma[n, 2]

    dphix = substract_and_fold(phix, padding)
    dphiy = substract_and_fold(phiy, padding)
    dphiz = substract_and_fold(phiz, padding)
    ddphix_dx = substract_and_fold(dphix_dx, padding)
    ddphiy_dy = substract_and_fold(dphiy_dy, padding)
    ddphiz_dz = substract_and_fold(dphiz_dz, padding)

    exp = experimental_map

    sim = np.zeros((k_dim, j_dim, i_dim), dtype=np.float64)
    for n in prange(n_dim):
        eps_n = epsilon[n]
        i_min, i_max, j_min, j_max, k_min, k_max = limits[n]
        for k in range(k_min, k_max + 1):
            k = (k - padding) % k_dim
            for j in range(j_min, j_max + 1):
                j = (j - padding) % j_dim
                for i in range(i_min, i_max + 1):
                    i = (i - padding) % i_dim
                    sim[k, j, i] += eps_n * dphix[n, i] * dphiy[n, j] * dphiz[n, k]

    sim_mean = np.mean(sim)
    sim_std = np.std(sim)
    sim = (sim - sim_mean) / sim_std
    cc = np.mean(sim * exp)

    num1 = np.zeros((n_dim, 3), dtype=np.float64)
    num2 = np.zeros((n_dim, 3), dtype=np.float64)
    for n in prange(n_dim):
        eps_n = epsilon[n]
        i_min, i_max, j_min, j_max, k_min, k_max = limits[n]
        for k in range(k_min, k_max + 1):
            k = (k - padding) % k_dim
            for j in range(j_min, j_max + 1):
                j = (j - padding) % j_dim
                for i in range(i_min, i_max + 1):
                    i = (i - padding) % i_dim
                    exp_val = exp[k, j, i]
                    sim_val = sim[k, j, i]
                    gx = eps_n * ddphix_dx[n, i] * dphiy[n, j] * dphiz[n, k]
                    gy = eps_n * dphix[n, i] * ddphiy_dy[n, j] * dphiz[n, k]
                    gz = eps_n * dphix[n, i] * dphiy[n, j] * ddphiz_dz[n, k]
                    num1[n, 0] += gx * exp_val
                    num1[n, 1] += gy * exp_val
                    num1[n, 2] += gz * exp_val
                    num2[n, 0] += gx * sim_val
                    num2[n, 1] += gy * sim_val
                    num2[n, 2] += gz * sim_val

    num2 *= cc
    den = sim_std * i_dim * j_dim * k_dim
    result = (num1 - num2) / den
    return result, cc
