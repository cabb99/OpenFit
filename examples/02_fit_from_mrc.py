"""Round-trip a density map through an MRC file and fit into it.

Demonstrates :meth:`Fit.save_mrc` and :meth:`Fit.from_mrc`. Requires the ``io``
extra (``mrcfile``)::

    pip install "openfit[io]"
    python examples/02_fit_from_mrc.py
"""

import tempfile
from pathlib import Path

import numpy as np

from openfit import Fit


def main(seed=0):
    rng = np.random.default_rng(seed)
    n = 6
    true_coords = rng.uniform(8, 22, size=(n, 3))
    sigma = np.full((n, 3), 2.0)
    epsilon = np.ones(n)

    # Build a target density from the ground-truth particles.
    template = Fit(np.zeros((30, 30, 30)), voxel_size=[1, 1, 1])
    template.set_coordinates(true_coords, sigma, epsilon)
    target_density = template.simulation_map()

    # Store it as an experimental map and write it to an MRC file.
    workdir = Path(tempfile.mkdtemp())
    target_path = workdir / "target.mrc"
    source = Fit(target_density, voxel_size=[1, 1, 1])
    source.save_mrc(str(target_path), experimental=True)

    # Load it back and fit a perturbed guess into it.
    fit = Fit.from_mrc(str(target_path))
    fit.set_coordinates(true_coords + rng.normal(scale=1.0, size=(n, 3)), sigma, epsilon)

    initial_cc = fit.corr_coef()
    for _ in range(50):
        grad = fit.dcorr_coef()[:, :3]
        fit.coordinates += (0.1 / np.abs(grad).max()) * grad
    final_cc = fit.corr_coef()

    # Save the fitted density next to the target.
    fit.save_mrc(str(workdir / "fitted.mrc"))
    return initial_cc, final_cc


if __name__ == "__main__":
    initial_cc, final_cc = main()
    print(f"initial correlation coefficient: {initial_cc:.4f}")
    print(f"final correlation coefficient:   {final_cc:.4f}")
