"""Flexible fitting of adenylate kinase into a cryo-EM-style density.

Adenylate kinase (AdK) undergoes a large open -> closed hinge motion. The *open*
structure (PDB 4AKE), as an OpenSMOG all-atom structure-based model, is flexibly
fitted into the density of the *closed* structure (PDB 1AKE).

This is the one-call version using the high-level ``openfit.Fit`` orchestrator:
``Fit.from_smog(...)`` builds the SMOG system, attaches the OpenFit density force
+ auto-updater, and ``refine()`` runs the fit.

Requirements (into the same conda env)::

    pip install OpenSMOG "openfit[openmm,io]"

Inputs (next to this script):
    4ake.AA.gro / .top / .xml   SMOG 2 all-atom model of open AdK (4AKE)
    1AKE.mrc                     target density (closed AdK)

Run::

    python run_fit.py                 # 50000 steps on CPU (~minutes)
    python run_fit.py --steps 10000   # shorter
"""

import argparse
from pathlib import Path

import numpy as np

from openfit import Fit

HERE = Path(__file__).resolve().parent


def main(steps=50000, update_interval=100, k=3200, sigma=1.5, platform="CPU"):
    fit = Fit.from_smog(
        HERE / "4ake.AA.gro",
        HERE / "4ake.AA.top",
        HERE / "4ake.AA.xml",
        HERE / "1AKE.mrc",
        sigma=sigma,
        k=k,
        update_interval=update_interval,
        platform=platform,
    )

    fit.save_map(str(HERE / "initial.mrc"))
    print(f"initial correlation coefficient: {fit.cc:.4f}", flush=True)

    result = fit.refine(steps=steps, record_interval=update_interval)

    fit.save(str(HERE / "final.pdb"))
    fit.save_map(str(HERE / "final.mrc"))
    np.savetxt(HERE / "correlation_history.txt", result["history"])
    print(f"final correlation coefficient:   {result['correlation']:.4f}", flush=True)
    return result["history"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--update-interval", type=int, default=100)
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()
    main(steps=args.steps, update_interval=args.update_interval, platform=args.platform)
