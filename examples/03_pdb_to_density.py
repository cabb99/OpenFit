"""Generate a density map from a PDB structure using MolScene.

Requires the ``pdb`` extra (`MolScene <https://github.com/cabb99/molscene>`_)::

    pip install "openfit[pdb]"
    python examples/03_pdb_to_density.py

A tiny inline structure is used so the example is self-contained; in practice
pass the path to your own PDB/CIF file to ``Scene.from_pdb`` / ``Scene.from_file``.
"""

import tempfile
from pathlib import Path

import numpy as np

from openfit import DensityMap

# A minimal 5-atom PDB (poly-glycine backbone fragment) with element symbols.
_SAMPLE_PDB = """\
ATOM      1  N   GLY A   1      11.104   6.134   7.123  1.00  0.00           N
ATOM      2  CA  GLY A   1      12.560   6.087   7.220  1.00  0.00           C
ATOM      3  C   GLY A   1      13.000   4.700   7.660  1.00  0.00           C
ATOM      4  O   GLY A   1      12.230   3.760   7.850  1.00  0.00           O
ATOM      5  N   GLY A   2      14.300   4.560   7.830  1.00  0.00           N
END
"""


def main():
    import molscene

    workdir = Path(tempfile.mkdtemp())
    pdb_path = workdir / "sample.pdb"
    pdb_path.write_text(_SAMPLE_PDB)

    scene = molscene.Scene.from_pdb(str(pdb_path))
    coords = scene.get_coordinates().to_numpy()
    masses = scene.compute_mass()["mass"].to_numpy()

    pad = 5.0
    fit = DensityMap.from_dimensions(coords.min(0) - pad, coords.max(0) + pad, voxel_size=[1, 1, 1])
    fit.set_coordinates(coords, sigma=np.full(coords.shape, 1.5), epsilon=masses)

    out = workdir / "structure_density.mrc"
    fit.save_mrc(str(out))
    return out


if __name__ == "__main__":
    out = main()
    print(f"wrote density map to {out}")
