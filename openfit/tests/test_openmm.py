"""Tests for the OpenMM integration (``add_force`` / ``update_force``).

These exercise the custom-force mechanism that injects the density-fitting
gradient into an OpenMM ``Context``: a ``CustomCompoundBondForce`` backed by a
tabulated function whose values are refreshed each step via
``updateParametersInContext`` (no Context rebuild). Skipped when OpenMM is not
installed.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from openfit import DensityMap  # noqa: E402  (import after importorskip)


class _ContextSim:
    """Minimal stand-in for an ``openmm.app.Simulation``.

    ``Fit.update_force`` / ``Fit.update_coordinates`` only touch
    ``simulation.context``, so we avoid pulling in ``openmm.app``.
    """

    def __init__(self, context):
        self.context = context


def _make_fit(n_particles):
    nx, ny, nz = 20, 20, 20
    rng = np.random.default_rng(0)
    coords = rng.random((n_particles, 3)) * 5.0  # angstrom, inside the box
    fit = DensityMap(rng.random((nz, ny, nx)), voxel_size=[1, 1, 1])
    fit.set_coordinates(coords, np.ones((n_particles, 3)), np.ones(n_particles))
    return fit


def test_add_force_registers_one_force():
    n = 5
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)

    fit = _make_fit(n)
    fit.add_force(system)

    assert system.getNumForces() == 1
    force = system.getForce(0)
    assert isinstance(force, openmm.CustomCompoundBondForce)
    assert force.getNumBonds() == n


def test_update_force_pushes_gradient_into_context():
    n = 5
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)

    fit = _make_fit(n)
    fit.add_force(system)

    integrator = openmm.VerletIntegrator(1.0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    positions = np.random.default_rng(1).random((n, 3)) * 0.5  # nm
    context.setPositions(positions)

    sim = _ContextSim(context)
    force_array = fit.update_force(sim, update_coordinates=True, k=1.0)

    # The returned gradient has one (x, y, z) row per particle...
    assert force_array.shape == (n, 3)
    assert np.isfinite(force_array).all()

    # ...and it was written into the tabulated function backing the force.
    tabulated = fit.force.getTabulatedFunction(0)
    stored = np.asarray(tabulated.getFunctionParameters()[2])
    assert np.allclose(stored, force_array.T.ravel())
