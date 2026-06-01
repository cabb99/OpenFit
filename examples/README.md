# OpenFit examples

Runnable scripts demonstrating the main workflows. Each defines a `main()`
function and is smoke-tested in `openfit/tests/test_examples.py`.

| Script | What it shows | Requires |
| --- | --- | --- |
| `01_fit_synthetic.py` | Fit particles into a synthetic density by following the correlation gradient | core only |
| `02_fit_from_mrc.py` | Save/load a density map and fit into it | `openfit[io]` (mrcfile) |
| `03_pdb_to_density.py` | Build a density map from a PDB structure via MolScene | `openfit[pdb]` (molscene) |
| `04_openmm_integration.py` | Drive an OpenMM simulation toward a target map | `openfit[openmm]` |

Run any example directly, e.g.:

```bash
python examples/01_fit_synthetic.py
```
