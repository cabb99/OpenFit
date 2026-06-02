API Reference
=============

.. currentmodule:: openfit

OpenFit exposes the :class:`DensityMap` density-fitting engine, the OpenMM
integration (:class:`DensityForce` / :class:`DensityForceUpdater`), and the
:func:`generate_rotations` helper.

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
