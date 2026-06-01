Getting Started
===============

OpenFit fits a molecular structure into a 3D density map (for example a cryo-EM
or cryo-ET map). Each particle is modelled as an anisotropic 3D Gaussian, and
the package computes the cross-correlation between the *simulated* density and an
*experimental* density map, along with its analytical gradient with respect to
the particle coordinates and Gaussian widths. That gradient can drive an
optimization directly, or be injected as a force into an OpenMM simulation.

Installation
------------

OpenFit requires ``numpy``, ``scipy`` and ``numba``. Optional features pull in
extra packages, declared as extras:

.. code-block:: bash

    # Core install
    pip install openfit

    # With OpenMM integration, density-map I/O, plotting, and the PDB workflow
    pip install "openfit[all]"

The available extras are ``openmm`` (molecular dynamics integration), ``io``
(``mrcfile`` + ``mdtraj``), ``pdb`` (the `MolScene
<https://github.com/cabb99/molscene>`_-based PDB/CIF workflow), and ``viz``
(``matplotlib``).

For development, create the conda environment and install in editable mode:

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yaml -n openfit_dev
    conda activate openfit_dev
    pip install -e . --no-deps

Quick start
-----------

Build a target density from a known set of particles, then recover their
positions from a perturbed guess by following the correlation gradient:

.. code-block:: python

    import numpy as np
    from openfit import Fit

    rng = np.random.default_rng(0)
    n = 6
    true_coords = rng.uniform(8, 22, size=(n, 3))
    sigma = np.full((n, 3), 2.0)
    epsilon = np.ones(n)

    # Generate a synthetic "experimental" map from the ground-truth particles.
    template = Fit(np.zeros((30, 30, 30)), voxel_size=[1, 1, 1])
    template.set_coordinates(true_coords, sigma, epsilon)
    experimental = template.simulation_map()

    # Fit, starting from a perturbed guess.
    fit = Fit(experimental, voxel_size=[1, 1, 1])
    fit.set_coordinates(true_coords + rng.normal(scale=1.0, size=(n, 3)), sigma, epsilon)
    print("initial cc:", fit.corr_coef())

    for _ in range(50):
        grad = fit.dcorr_coef()[:, :3]          # d(cc)/d(x, y, z)
        fit.coordinates += (0.1 / np.abs(grad).max()) * grad
    print("final cc:  ", fit.corr_coef())

This runnable script lives in ``examples/01_fit_synthetic.py``. See the
:doc:`user_guide` for loading real maps, saving results, and the OpenMM
integration.
