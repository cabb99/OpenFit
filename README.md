# OpenFit

[![GitHub Actions Build Status](https://github.com/cabb99/OpenFit/actions/workflows/CI.yaml/badge.svg)](https://github.com/cabb99/OpenFit/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/cabb99/OpenFit/branch/main/graph/badge.svg)](https://codecov.io/gh/cabb99/OpenFit/branch/main)

Molecular Density Fitting Using 3D Gaussian Functions

## Description

This project implements a molecular density fitting approach that optimizes the correlation between an experimental density map and a synthetic density map originating from molecular dynamics using 3D Gaussian functions. The algorithm generates a force in the direction that increases the correlation between the simulated and effective maps. It also allows adjusting anisotropically the spread parameter (sigma) in each direction.

## Features

- Efficient simulation of density maps using 3D Gaussian functions.
- Optimization of molecular coordinates and sigma parameters to improve fit.
- Calculation of the correlation coefficient to measure fit quality.
- Use of numerical and analytical methods to compute derivatives.
- Implementation optimized with Numba for high performance.

## Installation

OpenFit requires Python 3.10+, NumPy, SciPy and Numba.

```bash
# Core install
pip install openfit

# With OpenMM integration, density-map I/O, plotting, and the PDB workflow
pip install "openfit[all]"
```

Available extras: `openmm` (molecular dynamics integration), `io` (density-map
I/O with `mrcfile` + trajectory I/O with `mdtraj`), `pdb` (the
[MolScene](https://github.com/cabb99/molscene)-based PDB/CIF workflow), and
`viz` (`matplotlib`). `all` installs every extra.

## Usage

### Quick start

Fit a structure into a density map in one call. With a structure-based
(OpenSMOG) model:

```python
from openfit import Fit

fit = Fit.from_smog("model.AA.gro", "model.AA.top", "model.AA.xml", "target.mrc")
fit.refine(steps=50_000)     # bias an OpenMM MD run toward the density
print("correlation:", fit.cc)
fit.save("refined.pdb")
```

`Fit` also offers `from_amber(pdb, map)` and `from_system(topology, system, positions, map)`
(bring your own OpenMM force field). See [`examples/4ake/`](examples/4ake/) for a
complete, runnable flexible-fitting example (open→closed adenylate kinase).

### Scoring a density directly

Without any force field, use the `DensityMap` engine to rasterize particles,
score the map correlation, and optimize coordinates:

Each particle is an anisotropic 3D Gaussian set by its coordinates, per-axis
widths `sigma`, and a weight `epsilon` (often the atomic mass):

```python
import numpy as np
from openfit import DensityMap

n = 100
coordinates = np.random.default_rng(0).uniform(0, 40, size=(n, 3))
sigma = np.full((n, 3), 2.0)
epsilon = np.ones(n)                       # per-particle weight (e.g. mass)

dm = DensityMap(experimental_map, voxel_size=[1, 1, 1])   # a 3D numpy array
dm.set_coordinates(coordinates, sigma=sigma, epsilon=epsilon)
print("cc:", dm.correlation())
grad = dm.gradient()         # (n, 7): d(cc)/d(x, y, z, sx, sy, sz, epsilon)
dm.fit(n_iter=200)           # gradient ascent on the coordinates (no MD)
dm.save_mrc("simulated.mrc")
```

### Build a density map from a PDB structure

With the `pdb` extra ([MolScene](https://github.com/cabb99/molscene)):

```python
import molscene
from openfit import DensityMap

scene = molscene.Scene.from_pdb("structure.pdb")
dm = DensityMap.from_scene(scene)          # coordinates + atomic-mass weights
dm.save_mrc("structure_density.mrc")
```

See the [`examples/`](examples/) directory and the
[documentation](https://openfit.readthedocs.io/) for the full guide.

## Method

OpenFit adds a density-fitting potential to a standard force field,

$$ V = V_{ff} + V_{Fit}, \qquad V_{Fit} = k\,(1 - \text{c.c.}) $$

where the cross-correlation (c.c.) between the experimental and simulated
densities over the voxels $(i,j,k)$ is

$$ \text{c.c.} = \frac{\sum_{ijk} \rho_{\text{exp}}(i,j,k)\,\rho_{\text{sim}}(i,j,k)}{\sqrt{\sum_{ijk} \rho_{\text{exp}}(i,j,k)^2}\,\sqrt{\sum_{ijk} \rho_{\text{sim}}(i,j,k)^2}}. $$

The simulated density rasterizes each atom as an anisotropic 3D Gaussian; OpenFit
computes it and the analytical gradient of the correlation with respect to the
coordinates and Gaussian widths, and applies the corresponding force. The full
derivation — the Gaussian integral over each voxel (via the error function) and
the correlation derivatives — is in the
[user guide](https://openfit.readthedocs.io/en/latest/user_guide.html#method-and-derivation).

## Copyright

&copy; 2024-2026, Carlos Bueno

## Acknowledgements

Carlos Bueno was supported by the MolSSI Software Fellowship, under the mentorship of Jessica Nash. We thank AMD (Advanced Micro Devices, Inc.) for the donation of high-performance computing hardware and HPC resources. This project is also supported by the Center for Theoretical Biological Physics (NSF Grants PHY-2019745 and PHY-1522550), with additional support from the D.R. Bullard Welch Chair at Rice University (Grant No. C-0016 to PGW). Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
