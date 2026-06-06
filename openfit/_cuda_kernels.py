"""GPU density-correlation gradient implementation using numba.cuda."""

import logging
import math

import numpy as np

try:
    from numba import cuda

    CUDA_AVAILABLE = cuda.is_available()
except Exception:  # pragma: no cover - depends on numba.cuda toolchain
    cuda = None
    CUDA_AVAILABLE = False
    logging.debug("numba.cuda unavailable; the CUDA gradient backend is disabled.", exc_info=True)


if CUDA_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def _phi(X, mu, seff):
        return 0.5 * (1.0 + math.erf((X - mu) / seff))

    @cuda.jit(device=True, inline=True)
    def _dphi(X, mu, seff):
        z = (X - mu) / seff
        return -math.exp(-z * z) / (math.sqrt(math.pi) * seff)

    @cuda.jit
    def _precompute(coords, sigma, nx, ny, nz, vx, vy, vz, mult, limits, dp, dd):
        """One thread per atom: voxel limits + per-axis density/derivative factors.

        ``coords`` are already in the internal grid frame (``DensityMap.coordinates``), i.e.
        wrapped into ``[0, L)``; near-edge atoms are clamped (no PBC fold), matching the
        truncation used by the C++/CPU paths for a structure that sits inside the map box.
        """
        n = cuda.grid(1)
        if n >= coords.shape[0]:
            return
        s2 = math.sqrt(2.0)
        for axis in range(3):
            if axis == 0:
                ic = coords[n, 0]
                v = vx
                dim = nx
                sg = sigma[n, 0]
            elif axis == 1:
                ic = coords[n, 1]
                v = vy
                dim = ny
                sg = sigma[n, 1]
            else:
                ic = coords[n, 2]
                v = vz
                dim = nz
                sg = sigma[n, 2]
            seff = sg * s2
            half = mult * sg
            vmin = int(math.floor((ic - half) / v)) - 1
            vmax = int(math.ceil((ic + half) / v)) + 1
            if vmin < 0:
                vmin = 0
            if vmax > dim - 1:
                vmax = dim - 1
            limits[n, 2 * axis] = vmin
            limits[n, 2 * axis + 1] = vmax
            for off in range(vmax - vmin + 1):
                vv = vmin + off
                Xlo = vv * v
                Xhi = (vv + 1) * v
                dp[n, axis, off] = _phi(Xhi, ic, seff) - _phi(Xlo, ic, seff)
                dd[n, axis, off] = _dphi(Xhi, ic, seff) - _dphi(Xlo, ic, seff)

    @cuda.jit
    def _scatter(limits, dp, eps, nx, ny, nz, sim):
        n = cuda.blockIdx.x
        i0 = limits[n, 0]
        i1 = limits[n, 1]
        j0 = limits[n, 2]
        j1 = limits[n, 3]
        k0 = limits[n, 4]
        k1 = limits[n, 5]
        ni = i1 - i0 + 1
        nj = j1 - j0 + 1
        nk = k1 - k0 + 1
        if ni <= 0 or nj <= 0 or nk <= 0:
            return
        box = ni * nj * nk
        en = eps[n]
        for idx in range(cuda.threadIdx.x, box, cuda.blockDim.x):
            kk = idx // (ni * nj)
            rem = idx - kk * (ni * nj)
            jj = rem // ni
            ii = rem - jj * ni
            val = en * dp[n, 0, ii] * dp[n, 1, jj] * dp[n, 2, kk]
            flat = ((k0 + kk) * ny + (j0 + jj)) * nx + (i0 + ii)
            cuda.atomic.add(sim, flat, val)

    @cuda.jit
    def _sq(sim, mean, out):
        m = cuda.grid(1)
        if m < sim.size:
            d = sim[m] - mean
            out[m] = d * d

    @cuda.jit
    def _mul(sim, exp, out):
        m = cuda.grid(1)
        if m < sim.size:
            out[m] = sim[m] * exp[m]

    @cuda.jit
    def _gather(limits, dp, dd, eps, exp, sim, mean, std, nx, ny, nz, num1, num2):
        n = cuda.blockIdx.x
        i0 = limits[n, 0]
        i1 = limits[n, 1]
        j0 = limits[n, 2]
        j1 = limits[n, 3]
        k0 = limits[n, 4]
        k1 = limits[n, 5]
        ni = i1 - i0 + 1
        nj = j1 - j0 + 1
        nk = k1 - k0 + 1
        if ni <= 0 or nj <= 0 or nk <= 0:
            return
        box = ni * nj * nk
        en = eps[n]
        n1x = 0.0
        n1y = 0.0
        n1z = 0.0
        n2x = 0.0
        n2y = 0.0
        n2z = 0.0
        for idx in range(cuda.threadIdx.x, box, cuda.blockDim.x):
            kk = idx // (ni * nj)
            rem = idx - kk * (ni * nj)
            jj = rem // ni
            ii = rem - jj * ni
            flat = ((k0 + kk) * ny + (j0 + jj)) * nx + (i0 + ii)
            e = exp[flat]
            s = (sim[flat] - mean) / std
            gx = en * dd[n, 0, ii] * dp[n, 1, jj] * dp[n, 2, kk]
            gy = en * dp[n, 0, ii] * dd[n, 1, jj] * dp[n, 2, kk]
            gz = en * dp[n, 0, ii] * dp[n, 1, jj] * dd[n, 2, kk]
            n1x += gx * e
            n1y += gy * e
            n1z += gz * e
            n2x += gx * s
            n2y += gy * s
            n2z += gz * s
        cuda.atomic.add(num1, (n, 0), n1x)
        cuda.atomic.add(num1, (n, 1), n1y)
        cuda.atomic.add(num1, (n, 2), n1z)
        cuda.atomic.add(num2, (n, 0), n2x)
        cuda.atomic.add(num2, (n, 1), n2y)
        cuda.atomic.add(num2, (n, 2), n2z)

    _sum = cuda.reduce(lambda a, b: a + b)


class CudaGradientEvaluator:
    """Resident GPU evaluator for the density-correlation force gradient + cc.

    Build once per :class:`~openfit.DensityMap` (uploads the map/sigma/epsilon and allocates
    scratch); call with the internal-frame coordinates to get ``((n, 3) gradient, cc)``.
    """

    def __init__(self, density, multiplier=5, threadsperblock=128):
        if not CUDA_AVAILABLE:
            raise NotImplementedError(
                "device='cuda' requires a CUDA GPU with numba.cuda available "
                "(install a CUDA toolkit, e.g. `conda install cuda-nvcc`)."
            )
        self.nx, self.ny, self.nz = (int(density.n_voxels[0]), int(density.n_voxels[1]), int(density.n_voxels[2]))
        self.vx, self.vy, self.vz = (
            float(density.voxel_size[0]),
            float(density.voxel_size[1]),
            float(density.voxel_size[2]),
        )
        self.mult = float(multiplier)
        self.tpb = int(threadsperblock)
        self.nvox = self.nx * self.ny * self.nz
        self.maxl = 2 * int(density.padding) + 3
        self.N = int(density.coordinates.shape[0])
        self.exp_d = cuda.to_device(np.ascontiguousarray(density.experimental_map, dtype=np.float64).ravel())
        self.sigma_d = cuda.to_device(np.ascontiguousarray(density.sigma, dtype=np.float64))
        self.eps_d = cuda.to_device(np.ascontiguousarray(density.epsilon, dtype=np.float64))
        self.sim_d = cuda.device_array(self.nvox, dtype=np.float64)
        self.tmp_d = cuda.device_array(self.nvox, dtype=np.float64)
        self.limits_d = cuda.device_array((self.N, 6), dtype=np.int64)
        self.dp_d = cuda.device_array((self.N, 3, self.maxl), dtype=np.float64)
        self.dd_d = cuda.device_array((self.N, 3, self.maxl), dtype=np.float64)
        self.num1_d = cuda.device_array((self.N, 3), dtype=np.float64)
        self.num2_d = cuda.device_array((self.N, 3), dtype=np.float64)
        # source arrays this evaluator was built from (for staleness checks by force_gradient)
        self._src_sigma = density.sigma
        self._src_eps = density.epsilon
        self._src_exp = density.experimental_map

    def matches(self, density):
        """True if this evaluator is still valid for ``density`` (same map/sigma/epsilon/size)."""
        return (
            self._src_sigma is density.sigma
            and self._src_eps is density.epsilon
            and self._src_exp is density.experimental_map
            and self.N == int(density.coordinates.shape[0])
        )

    def __call__(self, coordinates):
        pos_d = cuda.to_device(np.ascontiguousarray(coordinates, dtype=np.float64))
        self.sim_d[:] = 0.0
        self.num1_d[:] = 0.0
        self.num2_d[:] = 0.0
        blocks = (self.N + self.tpb - 1) // self.tpb
        _precompute[blocks, self.tpb](
            pos_d,
            self.sigma_d,
            self.nx,
            self.ny,
            self.nz,
            self.vx,
            self.vy,
            self.vz,
            self.mult,
            self.limits_d,
            self.dp_d,
            self.dd_d,
        )
        _scatter[self.N, self.tpb](self.limits_d, self.dp_d, self.eps_d, self.nx, self.ny, self.nz, self.sim_d)

        vblocks = (self.nvox + 255) // 256
        mean = _sum(self.sim_d) / self.nvox
        _sq[vblocks, 256](self.sim_d, mean, self.tmp_d)
        std = math.sqrt(_sum(self.tmp_d) / self.nvox)
        _mul[vblocks, 256](self.sim_d, self.exp_d, self.tmp_d)
        cc = _sum(self.tmp_d) / (std * self.nvox)  # experimental map is zero-mean -> simplifies

        _gather[self.N, self.tpb](
            self.limits_d,
            self.dp_d,
            self.dd_d,
            self.eps_d,
            self.exp_d,
            self.sim_d,
            mean,
            std,
            self.nx,
            self.ny,
            self.nz,
            self.num1_d,
            self.num2_d,
        )
        num1 = self.num1_d.copy_to_host()
        num2 = self.num2_d.copy_to_host()
        grad = (num1 - num2 * cc) / (std * self.nvox)
        return grad, float(cc)
