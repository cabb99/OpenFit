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
(density-map I/O with ``mrcfile`` and trajectory I/O with ``mdtraj``), ``pdb``
(the `MolScene <https://github.com/cabb99/molscene>`_-based PDB/CIF workflow),
and ``viz`` (``matplotlib``). ``all`` installs every extra.

For development, create the conda environment and install in editable mode:

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yaml -n openfit_dev
    conda activate openfit_dev
    pip install -e . --no-deps

Quick start — flexible fitting in one call
------------------------------------------

The high-level :class:`~openfit.Fit` loads a structure + a map, refines, and
saves. With a structure-based (OpenSMOG) model:

.. code-block:: python

    from openfit import Fit

    fit = Fit.from_smog("model.AA.gro", "model.AA.top", "model.AA.xml", "target.mrc")
    fit.refine(steps=50_000)     # bias MD toward the density
    print("correlation:", fit.cc)
    fit.save("refined.pdb")

See ``examples/4ake/`` for a complete, runnable flexible-fitting example.

Quick start — the ``DensityMap`` engine
----------------------------------------

Without any OpenMM force field you can still score and optimize a density
directly. Build a target from known particles, then recover their positions from
a perturbed guess by following the correlation gradient:

.. code-block:: python

    import numpy as np
    from openfit import DensityMap

    rng = np.random.default_rng(0)
    n = 6
    true_coords = rng.uniform(8, 22, size=(n, 3))
    sigma = np.full((n, 3), 2.0)     # per-axis Gaussian widths
    epsilon = np.ones(n)             # per-particle weight (e.g. atomic mass)

    # Generate a synthetic "experimental" map from the ground-truth particles.
    template = DensityMap(np.zeros((30, 30, 30)), voxel_size=[1, 1, 1])
    template.set_coordinates(true_coords, sigma, epsilon)
    experimental = template.simulation_map()

    # Fit, starting from a perturbed guess.
    dm = DensityMap(experimental, voxel_size=[1, 1, 1])
    dm.set_coordinates(true_coords + rng.normal(scale=1.0, size=(n, 3)), sigma, epsilon)
    print("initial cc:", dm.correlation())

    for _ in range(50):
        grad = dm.gradient()[:, :3]          # d(cc)/d(x, y, z)
        dm.coordinates += (0.1 / np.abs(grad).max()) * grad
    print("final cc:  ", dm.correlation())

This runnable script lives in ``examples/01_fit_synthetic.py``. See the
:doc:`user_guide` for loading real maps, saving results, the OpenMM integration,
and the method's derivation.

Quick start — the command line
------------------------------

Installing OpenFit provides an ``openfit`` command:

.. code-block:: bash

    # SMOG structure-based model
    openfit refine --smog model.AA.gro model.AA.top model.AA.xml target.mrc \
        -o refined.pdb --steps 50000

    # all-atom PDB (Amber14 + implicit solvent)
    openfit refine --pdb model.pdb target.mrc -o refined.pdb

    # or from a YAML config (needs the ``cli`` extra)
    openfit run config.yaml

The YAML mirrors the flags:

.. code-block:: yaml

    smog: [model.AA.gro, model.AA.top, model.AA.xml]   # or: pdb: model.pdb
    map: target.mrc
    output: refined.pdb
    output_map: fitted.mrc        # optional
    steps: 50000
    k: 3200
    update_interval: 100
