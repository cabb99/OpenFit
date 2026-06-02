"""Fit particles into a synthetic density map (no optional dependencies).

Build a target density from a known set of particles, then recover their
positions from a perturbed guess by following the analytical correlation
gradient. This is the example exercised end-to-end by the test suite.

Run with::

    python examples/01_fit_synthetic.py
"""

import numpy as np

from openfit import DensityMap


def main(n_steps=50, seed=0):
    rng = np.random.default_rng(seed)
    n = 6
    true_coords = rng.uniform(8, 22, size=(n, 3))
    sigma = np.full((n, 3), 2.0)
    epsilon = np.ones(n)

    # Generate a synthetic "experimental" map from the ground-truth particles.
    template = DensityMap(np.zeros((30, 30, 30)), voxel_size=[1, 1, 1])
    template.set_coordinates(true_coords, sigma, epsilon)
    experimental = template.simulation_map()

    # Fit, starting from a perturbed guess.
    fit = DensityMap(experimental, voxel_size=[1, 1, 1])
    fit.set_coordinates(true_coords + rng.normal(scale=1.0, size=(n, 3)), sigma, epsilon)

    initial_cc = fit.correlation()
    for _ in range(n_steps):
        grad = fit.gradient()[:, :3]  # d(cc)/d(x, y, z)
        fit.coordinates += (0.1 / np.abs(grad).max()) * grad
    final_cc = fit.correlation()

    return initial_cc, final_cc


if __name__ == "__main__":
    initial_cc, final_cc = main()
    print(f"initial correlation coefficient: {initial_cc:.4f}")
    print(f"final correlation coefficient:   {final_cc:.4f}")
