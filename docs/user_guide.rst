User Guide
==========

This page covers the main OpenFit workflows. For the full method (the Gaussian
density model and the correlation-coefficient derivatives) see the derivation in
the project ``README``.

Creating a ``Fit``
------------------

A :class:`~openfit.Fit` is built around an experimental density map. The map is
stored normalized to zero mean and unit standard deviation. There are three ways
to create one:

.. code-block:: python

    import numpy as np
    from openfit import Fit

    # 1. From an in-memory array (z, y, x ordering), with a voxel size in Angstrom.
    fit = Fit(experimental_map, voxel_size=[1.0, 1.0, 1.0], origin=[0, 0, 0])

    # 2. From an MRC/CCP4 file (voxel size and origin are read from the header).
    fit = Fit.from_mrc("map.mrc", cutoff_min=0.0)

    # 3. From coordinate bounds, when you only need an empty grid to draw into.
    fit = Fit.from_dimensions(min_coords=[0, 0, 0], max_coords=[40, 40, 40], voxel_size=[2, 2, 2])

Setting particles
-----------------

Particles are defined by their coordinates (``n x 3``, in Angstrom), anisotropic
Gaussian widths ``sigma`` (``n x 3``), and per-particle weights ``epsilon``
(``n``; often the atomic mass):

.. code-block:: python

    fit.set_coordinates(coordinates, sigma=np.full((n, 3), 2.0), epsilon=masses)

``sigma`` and ``epsilon`` are remembered between calls, so subsequent
``set_coordinates(new_coords)`` reuse them.

Simulating and scoring
----------------------

.. code-block:: python

    density = fit.simulation_map()      # simulated density on the experimental grid
    cc = fit.corr_coef()                # cross-correlation with the experimental map
    grad = fit.dcorr_coef()             # (n, 7): d(cc)/d(x, y, z, sx, sy, sz, epsilon)

``dcorr_coef`` is the analytical gradient computed by a Numba kernel; it is
validated against finite differences (``dcorr_coef_numerical``) and a pure-NumPy
reference (``dcorr_coef_numpy``) in the test suite.

Saving a density map
--------------------

.. code-block:: python

    fit.save_mrc("simulated.mrc")                    # the simulated density
    fit.save_mrc("experimental.mrc", experimental=True)

OpenMM integration
------------------

OpenFit can bias a running OpenMM simulation toward higher map correlation. It
adds a ``CustomCompoundBondForce`` backed by a tabulated per-particle force, and
refreshes that table each time you call :meth:`~openfit.Fit.update_force`
(no Context rebuild):

.. code-block:: python

    fit.add_force(system)                 # add the fitting force to an OpenMM System
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    for _ in range(n_cycles):
        simulation.step(100)
        fit.update_force(simulation, k=3200)   # recompute gradient, push into the Context

``update_force`` reads the current positions from the simulation, recomputes the
correlation gradient, scales it by ``k``, and writes it into the force. See
``examples/04_openmm_integration.py`` for a complete minimal loop.

Working from a PDB/CIF structure
--------------------------------

With the ``pdb`` extra (`MolScene <https://github.com/cabb99/molscene>`_) you can
go straight from a structure file to a density map:

.. code-block:: python

    import numpy as np
    import molscene
    from openfit import Fit

    scene = molscene.Scene.from_pdb("structure.pdb")
    coords = scene.get_coordinates().to_numpy()
    masses = scene.compute_mass()["mass"].to_numpy()

    pad = 5.0
    fit = Fit.from_dimensions(coords.min(0) - pad, coords.max(0) + pad, voxel_size=[2, 2, 2])
    fit.set_coordinates(coords, sigma=np.full(coords.shape, 2.0), epsilon=masses)
    fit.save_mrc("structure_density.mrc")

See ``examples/03_pdb_to_density.py``.
