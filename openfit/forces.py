"""OpenMM integration — the *"python"* density-fitting force backend.

The correlation gradient is computed in Python (the CPU/GPU kernels on
:class:`~openfit.DensityMap`) and injected into OpenMM through a tabulated
``CustomCompoundBondForce`` that is refreshed every few steps. This is the
portable default backend.

A native C++/CUDA OpenMM force *plugin* (the ``"native"`` backend) is planned for
the performance phase; it will expose the same ``add_to`` / ``attach`` role so it
drops in without changing the high-level API.
"""

import numpy as np


class DensityForce:
    """A density-correlation biasing force for OpenMM (python backend).

    Wraps a ``CustomCompoundBondForce`` whose per-particle force is read from a
    tabulated function. Build it onto a ``System`` with :meth:`add_to` (before
    the ``Context`` exists), then refresh it during the simulation with
    :meth:`update` — or, more conveniently, let a :class:`DensityForceUpdater` do it.

    Parameters
    ----------
    density_map : openfit.DensityMap
        The engine that computes the correlation gradient.
    k : float, optional
        Force constant scaling the correlation gradient. Default 3200.
    """

    def __init__(self, density_map, k=3200):
        self.density = density_map
        self.k = k
        self._force = None

    @property
    def force(self):
        """The underlying ``openmm.CustomCompoundBondForce`` (None until added)."""
        return self._force

    def add_to(self, system):
        """Build the force and add it to an OpenMM ``System``.

        Must be called before a ``Context`` is created (OpenMM does not allow
        adding forces to an already-initialized context).

        Returns
        -------
        openmm.CustomCompoundBondForce
            The force that was added (also stored on :attr:`force`).
        """
        import openmm

        n_particles = system.getNumParticles()
        force = openmm.CustomCompoundBondForce(1, "-k(i,0)*x1-k(i,1)*y1-k(i,2)*z1")
        table = openmm.Discrete2DFunction(n_particles, 3, np.zeros((n_particles, 3)).T.flatten())
        force.addTabulatedFunction("k", table)
        force.addPerBondParameter("i")
        for i in range(n_particles):
            force.addBond([i], [i])
        force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(force)
        self._force = force
        return force

    def update(self, context, k=None, force_array=None):
        """Recompute the gradient and push it into the running ``Context``.

        Parameters
        ----------
        context : openmm.Context
            The context whose force parameters are updated in place.
        k : float, optional
            Override the force constant for this update (defaults to ``self.k``).
        force_array : numpy.ndarray, optional
            Explicit ``(n, 3)`` force values, bypassing the gradient computation.

        Returns
        -------
        numpy.ndarray
            The ``(n, 3)`` force array written to the context.
        """
        if self._force is None:
            raise RuntimeError("DensityForce.add_to(system) must be called before update().")
        if force_array is None:
            scale = self.k if k is None else k
            force_array = scale * self.density.gradient()[:, :3]
        table = self._force.getTabulatedFunction(0)
        params = table.getFunctionParameters()
        params[2] = force_array.T.ravel()
        table.setFunctionParameters(*params)
        self._force.updateParametersInContext(context)
        return force_array


class DensityForceUpdater:
    """Refresh a :class:`DensityForce` from the simulation state during MD.

    It uses OpenMM's reporter hook (``describeNextReport`` / ``report``) so you
    append it to ``simulation.reporters``, but it is **not** a reporter in the
    usual sense: a real OpenMM reporter only *observes* state, whereas this one
    *mutates* the context. Every ``interval`` steps it reads the current
    positions, updates the :class:`~openfit.DensityMap` coordinates, recomputes
    the correlation gradient, and pushes it into the force — removing the need
    for a manual per-step update loop.

    (The name ``DensityReporter`` is intentionally reserved for a future
    read-only reporter that logs the correlation coefficient / fit energy.)

    Parameters
    ----------
    density_force : DensityForce
        The force to refresh (already added to the system).
    interval : int, optional
        Steps between updates. Default 50.
    k : float, optional
        Force constant override passed to :meth:`DensityForce.update`.
    """

    def __init__(self, density_force, interval=50, k=None):
        self.density_force = density_force
        self.interval = int(interval)
        self.k = k

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return {"steps": steps, "periodic": False, "include": ["positions"]}

    def report(self, simulation, state):
        import openmm

        coordinates = np.array(state.getPositions().value_in_unit(openmm.unit.angstrom))
        self.density_force.density.set_coordinates(coordinates)
        self.density_force.update(simulation.context, k=self.k)
