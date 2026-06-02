"""Bias an OpenMM simulation toward a target density map.

Requires the ``openmm`` extra::

    pip install "openfit[openmm]"
    python examples/04_openmm_integration.py

A handful of free particles are driven by the OpenFit correlation force. A
``DensityForceUpdater`` refreshes the force's tabulated per-atom gradient every few
MD steps (no manual loop, no Context rebuild).
"""

import numpy as np

from openfit import DensityForce, DensityForceUpdater, DensityMap


def main(total_steps=200, update_every=50, seed=0):
    import openmm
    from openmm import app, unit

    n = 5
    rng = np.random.default_rng(seed)

    # Synthetic target density from a known arrangement of particles.
    target = DensityMap(np.zeros((20, 20, 20)), voxel_size=[1, 1, 1])
    target.set_coordinates(rng.uniform(5, 15, size=(n, 3)), np.full((n, 3), 2.0), np.ones(n))

    density = DensityMap(target.simulation_map(), voxel_size=[1, 1, 1])
    density.set_coordinates(rng.uniform(5, 15, size=(n, 3)), np.full((n, 3), 2.0), np.ones(n))

    # Minimal OpenMM system: n free particles in the map's periodic box.
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in density.periodic_vectors()])
    for _ in range(n):
        system.addParticle(12.0)

    # Add the density-fitting force (before the Context is created).
    force = DensityForce(density, k=3200)
    force.add_to(system)

    topology = app.Topology()
    residue = topology.addResidue("UNK", topology.addChain())
    for i in range(n):
        topology.addAtom(f"P{i}", app.Element.getBySymbol("C"), residue)

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions((density.coordinates / 10.0) * unit.nanometer)  # A -> nm

    # The updater refreshes the force every `update_every` steps.
    simulation.reporters.append(DensityForceUpdater(force, interval=update_every))

    cc_start = density.correlation()
    simulation.step(total_steps)
    cc_end = density.correlation()

    return cc_start, cc_end


if __name__ == "__main__":
    cc_start, cc_end = main()
    print(f"correlation coefficient before MD: {cc_start:.4f}")
    print(f"correlation coefficient after MD:  {cc_end:.4f}")
