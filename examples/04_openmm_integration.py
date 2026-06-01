"""Bias an OpenMM simulation toward a target density map.

Requires the ``openmm`` extra::

    pip install "openfit[openmm]"
    python examples/04_openmm_integration.py

A handful of free particles are driven by the OpenFit correlation force. The
force's tabulated per-particle gradient is refreshed every few MD steps via
:meth:`Fit.update_force`, without rebuilding the OpenMM ``Context``.
"""

import numpy as np

from openfit import Fit


def main(total_steps=200, update_every=50, seed=0):
    import openmm
    from openmm import app, unit

    n = 5
    rng = np.random.default_rng(seed)

    # Synthetic target density from a known arrangement of particles.
    target = Fit(np.zeros((20, 20, 20)), voxel_size=[1, 1, 1])
    target.set_coordinates(rng.uniform(5, 15, size=(n, 3)), np.full((n, 3), 2.0), np.ones(n))

    fit = Fit(target.simulation_map(), voxel_size=[1, 1, 1])
    fit.set_coordinates(rng.uniform(5, 15, size=(n, 3)), np.full((n, 3), 2.0), np.ones(n))

    # Minimal OpenMM system: n free particles in the map's periodic box.
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in fit.periodic_vectors()])
    for _ in range(n):
        system.addParticle(12.0)
    fit.add_force(system)

    topology = app.Topology()
    residue = topology.addResidue("UNK", topology.addChain())
    for i in range(n):
        topology.addAtom(f"P{i}", app.Element.getBySymbol("C"), residue)

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions((fit.coordinates / 10.0) * unit.nanometer)  # A -> nm

    fit.update_force(simulation, k=3200)
    cc_start = fit.corr_coef()
    for _ in range(total_steps // update_every):
        simulation.step(update_every)
        fit.update_force(simulation, k=3200)
    cc_end = fit.corr_coef()

    return cc_start, cc_end


if __name__ == "__main__":
    cc_start, cc_end = main()
    print(f"correlation coefficient before MD: {cc_start:.4f}")
    print(f"correlation coefficient after MD:  {cc_end:.4f}")
