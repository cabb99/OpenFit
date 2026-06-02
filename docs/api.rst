API Reference
=============

.. currentmodule:: openfit

OpenFit's public surface: the high-level :class:`Fit` orchestrator, the
:class:`DensityMap` density-fitting engine, the OpenMM integration
(:class:`DensityForce` / :class:`DensityForceUpdater`), and the
:func:`generate_rotations` helper.

The ``Fit`` orchestrator
------------------------

.. autoclass:: Fit
   :members:
   :member-order: bysource

The ``DensityMap`` engine
-------------------------

.. autoclass:: DensityMap
   :members:
   :undoc-members:
   :member-order: bysource

OpenMM integration
------------------

.. autoclass:: DensityForce
   :members:

.. autoclass:: DensityForceUpdater
   :members:

Rotations
---------

.. autofunction:: generate_rotations
