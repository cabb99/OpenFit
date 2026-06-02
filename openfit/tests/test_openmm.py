"""Tests for the OpenMM layer (``DensityForce`` / ``DensityForceUpdater``).

These exercise the "python" force backend: a ``CustomCompoundBondForce`` backed
by a tabulated function whose values are refreshed from the Python-computed
gradient via ``updateParametersInContext`` (no Context rebuild). The reporter
drives that refresh during MD. Skipped when OpenMM is not installed.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from openfit import DensityForce, DensityMap, DensityForceUpdater  # noqa: E402  (after importorskip)


def _make_density(n_particles):
    nx, ny, nz = 20, 20, 20
    rng = np.random.default_rng(0)
    coords = rng.random((n_particles, 3)) * 5.0  # angstrom, inside the box
    dm = DensityMap(rng.random((nz, ny, nx)), voxel_size=[1, 1, 1])
    dm.set_coordinates(coords, np.ones((n_particles, 3)), np.ones(n_particles))
    return dm


def _make_system(n):
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    return system


def test_density_force_add_to_registers_one_force():
    n = 5
    system = _make_system(n)
    DensityForce(_make_density(n)).add_to(system)

    assert system.getNumForces() == 1
    force = system.getForce(0)
    assert isinstance(force, openmm.CustomCompoundBondForce)
    assert force.getNumBonds() == n


def test_density_force_update_pushes_gradient_into_context():
    n = 5
    dm = _make_density(n)
    system = _make_system(n)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in dm.periodic_vectors()])

    force = DensityForce(dm, k=1.0)
    force.add_to(system)

    integrator = openmm.VerletIntegrator(1.0)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.random.default_rng(1).random((n, 3)) * 0.5)  # nm

    force_array = force.update(context)
    assert force_array.shape == (n, 3)
    assert np.isfinite(force_array).all()

    stored = np.asarray(force.force.getTabulatedFunction(0).getFunctionParameters()[2])
    assert np.allclose(stored, force_array.T.ravel())


def test_density_force_update_requires_add_to():
    force = DensityForce(_make_density(5))
    with pytest.raises(RuntimeError):
        force.update(context=None)


def test_density_reporter_refreshes_force_during_md():
    from openmm import app, unit

    n = 5
    dm = _make_density(n)
    system = _make_system(n)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in dm.periodic_vectors()])

    force = DensityForce(dm, k=3200)
    force.add_to(system)

    topology = app.Topology()
    residue = topology.addResidue("UNK", topology.addChain())
    for i in range(n):
        topology.addAtom(f"P{i}", app.Element.getBySymbol("C"), residue)

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds)
    simulation = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("Reference"))
    simulation.context.setPositions((dm.coordinates / 10.0) * unit.nanometer)
    simulation.reporters.append(DensityForceUpdater(force, interval=10))

    simulation.step(25)  # triggers the reporter at least twice

    stored = np.asarray(force.force.getTabulatedFunction(0).getFunctionParameters()[2])
    assert np.isfinite(stored).all()
    assert np.any(stored != 0.0)  # the reporter wrote a non-trivial gradient
