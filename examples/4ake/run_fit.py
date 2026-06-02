"""Flexible fitting of adenylate kinase into a cryo-EM-style density.

Adenylate kinase (AdK) undergoes a large open -> closed hinge motion. Here the
*open* structure (PDB 4AKE), represented by an OpenSMOG all-atom structure-based
model, is flexibly fitted into the density of the *closed* structure (PDB 1AKE).

OpenSMOG provides the structure-based (Go-like) force field that keeps the
protein physically reasonable, while OpenFit adds a force that pulls the atoms
toward higher correlation with the target density. The OpenFit force is a
tabulated per-atom gradient that is refreshed every few MD steps via
``Fit.update_force`` (no OpenMM Context rebuild).

Requirements (into the same conda env)::

    pip install OpenSMOG "openfit[openmm,io]"

Inputs (next to this script):
    4ake.AA.gro / .top / .xml   SMOG 2 all-atom model of open AdK (4AKE)
    1AKE.mrc                     target density (closed AdK)

Run::

    python run_fit.py                       # default: 500 fitting cycles on CPU
    python run_fit.py --fitting-iterations 100
"""

import argparse
from pathlib import Path

import numpy as np
import openmm
from OpenSMOG import SBM

from openfit import DensityMap

HERE = Path(__file__).resolve().parent

# First letter of each SMOG all-atom atom name is its element.
ELEMENT_MASS = {"C": 12.0, "N": 14.0, "O": 16.0, "S": 32.0, "H": 1.0, "P": 31.0}


def _positions_angstrom(simulation):
    state = simulation.context.getState(getPositions=True)
    return np.array(state.getPositions().value_in_unit(openmm.unit.angstrom))


def main(
    equilibration_steps=1000,
    fitting_iterations=500,
    steps_per_iteration=100,
    k=3200,
    sigma_width=1.5,
    platform="CPU",
    out_folder="output_4ake_AA",
):
    # --- 1. Structure-based model of open AdK --------------------------------
    sbm = SBM(name="4ake", time_step=0.002, collision_rate=1.0, r_cutoff=1.2, temperature=0.5, warn=False)
    sbm.setup_openmm(platform=platform, GPUindex="default")
    sbm.saveFolder(str(HERE / out_folder))
    sbm.loadSystem(
        Grofile=str(HERE / "4ake.AA.gro"),
        Topfile=str(HERE / "4ake.AA.top"),
        Xmlfile=str(HERE / "4ake.AA.xml"),
    )

    # --- 2. Target density and the OpenFit fitting force ---------------------
    fit = DensityMap.from_mrc(str(HERE / "1AKE.mrc"))
    sbm.system.setDefaultPeriodicBoxVectors(*fit.periodic_vectors())
    fit.add_force(sbm.system)

    # Drop the COM-motion remover so the fitting force can translate the protein.
    for i, force in reversed(list(enumerate(sbm.system.getForces()))):
        if isinstance(force, openmm.CMMotionRemover):
            sbm.system.removeForce(i)

    sbm.loaded = False
    sbm.createSimulation()
    sbm.createReporters(trajectory=True, energies=True, energy_components=True, interval=steps_per_iteration * 10)

    # --- 3. Per-atom Gaussian parameters -------------------------------------
    atom_names = sbm.Gro.atomNames
    n_atoms = len(atom_names)
    sigma = np.ones((n_atoms, 3)) * sigma_width
    epsilon = np.array([ELEMENT_MASS[name[0]] for name in atom_names], dtype=float)

    fit.set_coordinates(_positions_angstrom(sbm.simulation), sigma, epsilon)
    fit.save_mrc(str(HERE / "initial.mrc"))
    print(f"initial correlation coefficient: {fit.correlation():.4f}", flush=True)

    # --- 4. Equilibrate, then iteratively fit --------------------------------
    sbm.run(nsteps=equilibration_steps, report=True, interval=equilibration_steps)

    history = []
    for i in range(fitting_iterations):
        # update_force reads the current positions, recomputes the correlation
        # gradient, scales it by k, and writes it into the tabulated force.
        fit.update_force(sbm.simulation, k=k)
        cc = fit.correlation()
        history.append(cc)
        if i % 10 == 0:
            print(f"iter {i:4d}  cc = {cc:.4f}", flush=True)
        sbm.simulation.step(steps_per_iteration)

    fit.set_coordinates(_positions_angstrom(sbm.simulation), sigma, epsilon)
    fit.save_mrc(str(HERE / "final.mrc"))
    final_cc = fit.correlation()
    print(f"final correlation coefficient:   {final_cc:.4f}", flush=True)

    np.savetxt(HERE / "correlation_history.txt", np.asarray(history))
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equilibration-steps", type=int, default=1000)
    parser.add_argument("--fitting-iterations", type=int, default=500)
    parser.add_argument("--steps-per-iteration", type=int, default=100)
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()
    main(
        equilibration_steps=args.equilibration_steps,
        fitting_iterations=args.fitting_iterations,
        steps_per_iteration=args.steps_per_iteration,
        platform=args.platform,
    )
