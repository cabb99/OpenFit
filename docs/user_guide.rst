User Guide
==========

OpenFit fits a molecular structure into a 3D density map. It supplies a
*density* energy term; the physical force field (bonds, non-bonded, …) comes
from an OpenMM system you provide. There are two layers:

- :class:`~openfit.Fit` — the high-level orchestrator: load a system + a map,
  :meth:`~openfit.Fit.refine`, and save.
- :class:`~openfit.DensityMap` — the engine: rasterize particles to a density,
  score the correlation, and compute its gradient (no OpenMM required).

High-level flexible fitting with ``Fit``
----------------------------------------

``Fit`` wires a density-fitting force into an OpenMM simulation and refreshes it
during the run. Build it from a structure-based model, an Amber model, or your
own OpenMM ``System``:

.. code-block:: python

    from openfit import Fit

    # Structure-based (OpenSMOG) model -> fit into a target map
    fit = Fit.from_smog("model.AA.gro", "model.AA.top", "model.AA.xml", "target.mrc")
    result = fit.refine(steps=50_000)     # bias MD toward the density
    print("correlation:", fit.cc)
    fit.save("refined.pdb")
    fit.save_map("fitted.mrc")

Other constructors:

.. code-block:: python

    # All-atom MM from a (clean) PDB
    fit = Fit.from_amber("model.pdb", "target.mrc")        # Amber14 + implicit solvent
    fit = Fit.from_charmm("model.pdb", "target.mrc")       # CHARMM36 (bundled with OpenMM)

    # Generate a SMOG structure-based model from a PDB/CIF with SMOG 2 (external tool)
    fit = Fit.from_smog_structure("model.pdb", "target.mrc", model="AA")

    # Coarse-grained OpenAWSEM model (needs the `awsem` extra)
    fit = Fit.from_awsem("model.pdb", "target.mrc", chains="A")

    # Bring your own OpenMM system/topology/positions
    fit = Fit.from_system(topology, system, positions, "target.mrc",
                          platform="CUDA", k=6400, update_interval=50)

``from_amber``/``from_charmm`` are all-atom and assume a clean structure;
``from_smog`` / ``from_smog_structure`` build on SMOG 2 / OpenSMOG [OpenSMOG]_
(``from_smog_structure`` needs SMOG 2 installed, https://smog-server.org);
``from_awsem`` is coarse-grained (Cα/Cβ/O beads) and needs ``openawsem``.

:meth:`~openfit.Fit.refine` returns ``{"correlation", "steps", "history"}``;
:attr:`~openfit.Fit.cc` recomputes the correlation from the current positions.
Pass ``backend="python"`` (default); ``backend="native"`` (a C++/CUDA OpenMM
plugin) is reserved for the performance phase.

Rigid-body placement (optional)
--------------------------------

By default OpenFit assumes the structure is already aligned to the map. 
Because flexible fitting refines only locally, the structure should start roughly inside the density.
If it does not, ``Fit`` can optionally run a rigid-body search first (``rigid_search=...``) and begin the
refinement from the best placement. The search has two stages: a coarse scan over evenly
spaced orientations centered on the map's density-weighted centroid, 
followed by a **local refinement** (small random rotation + translation hill-climb) of the best few poses:

.. code-block:: python

    fit = Fit.from_smog(..., rigid_search=True)          # dock, then refine
    fit = Fit.from_smog(..., rigid_search={"n_rotations": 300, "refine_iters": 200})

    fit.dock(n_rotations=300)                            # or dock explicitly, later
    fit.refine(steps=50_000)

The same search is available on a bare engine via
:meth:`~openfit.DensityMap.rigid_fit`, which returns the best pose
(``{"coordinates", "rotation", "translation", "cc"}``) and leaves the structure
there. With even sampling (``n_rotations=300``) it recovers asymmetric structures
to ~1 Å; it is a coarse global pre-placement and assumes the structure fits
inside the map box.

The ``DensityMap`` engine
-------------------------

A :class:`~openfit.DensityMap` holds an experimental density (stored normalized
to zero mean and unit standard deviation). Create one in any of four ways:

.. code-block:: python

    import numpy as np
    from openfit import DensityMap

    # 1. From an in-memory array (z, y, x order), voxel size in Angstrom.
    dm = DensityMap(experimental_map, voxel_size=[1.0, 1.0, 1.0], origin=[0, 0, 0])

    # 2. From an MRC/CCP4 file (voxel size + origin read from the header).
    dm = DensityMap.from_mrc("map.mrc", cutoff_min=0.0)

    # 3. From coordinate bounds, when you only need an empty grid to draw into.
    dm = DensityMap.from_dimensions(min_coords=[0, 0, 0], max_coords=[40, 40, 40], voxel_size=[2, 2, 2])

    # 4. From a MolScene structure (sets coordinates + atomic-mass weights).
    dm = DensityMap.from_scene(scene)

Particles are defined by coordinates (``n x 3``, Angstrom), anisotropic Gaussian
widths ``sigma`` (``n x 3``), and per-particle weights ``epsilon`` (``n``, often
the atomic mass). ``sigma`` and ``epsilon`` are remembered between calls:

.. code-block:: python

    dm.set_coordinates(coordinates, sigma=np.full((n, 3), 2.0), epsilon=masses)

    density = dm.simulation_map()   # simulated density on the experimental grid
    cc = dm.correlation()           # cross-correlation with the experimental map
    grad = dm.gradient()            # (n, 7): d(cc)/d(x, y, z, sx, sy, sz, epsilon)

    dm.fit(n_iter=200)              # gradient ascent on coordinates (no MD)
    dm.save_mrc("simulated.mrc")

``gradient`` is the analytical gradient from a Numba kernel; it is validated
against finite differences (``gradient_numerical``) and a pure-NumPy reference
(``gradient_numpy``) in the test suite. ``simulation_map`` and ``gradient`` take
a ``device="cpu"`` argument; a ``"cuda"`` kernel is planned for the performance
phase.

OpenMM integration internals
-----------------------------

``Fit`` uses these under the hood; reach for them directly only if you manage the
simulation yourself. :class:`~openfit.DensityForce` wraps a
``CustomCompoundBondForce`` whose per-particle force is read from a tabulated
function, and :class:`~openfit.DensityForceUpdater` refreshes it during MD:

.. code-block:: python

    from openfit import DensityForce, DensityForceUpdater

    force = DensityForce(dm, k=6400)
    force.add_to(system)                 # before the Context is created
    # ... build simulation ...
    simulation.reporters.append(DensityForceUpdater(force, interval=50))

The updater uses OpenMM's reporter hook but, unlike a true reporter, it *mutates*
the context (it writes the recomputed gradient into the force). The name
``DensityReporter`` is reserved for a future read-only metrics logger.

Working from a PDB/CIF structure
--------------------------------

With the ``pdb`` extra (`MolScene <https://github.com/cabb99/molscene>`_) you can
go straight from a structure file to a density map:

.. code-block:: python

    import molscene
    from openfit import DensityMap

    scene = molscene.Scene.from_pdb("structure.pdb")
    dm = DensityMap.from_scene(scene)          # coordinates + atomic-mass weights
    dm.save_mrc("structure_density.mrc")

See ``examples/03_pdb_to_density.py``.

Method and derivation
======================

Energy
------

OpenFit implements :math:`V_{Fit}`, a potential added to an OpenMM force field:

.. math::

    V = V_{ff} + V_{Fit}

defined through the correlation coefficient (c.c.) between the experimental and
simulated densities at each voxel :math:`(i, j, k)`:

.. math::

    V_{Fit} = k \, (1 - \text{c.c.})

.. math::

    \text{c.c.} = \frac{\sum_{ijk} \rho_{\text{exp}}(i,j,k)\, \rho_{\text{sim}}(i,j,k)}
    {\sqrt{\sum_{ijk} \rho_{\text{exp}}(i,j,k)^2}\; \sqrt{\sum_{ijk} \rho_{\text{sim}}(i,j,k)^2}}

In practice both densities are mean-subtracted and scaled to unit standard
deviation before the sums (the experimental map when the :class:`~openfit.DensityMap`
is created, the simulated map inside :meth:`~openfit.DensityMap.correlation`), so
this equals the Pearson correlation coefficient.

The simulated density :math:`\rho_{\text{sim}}(i,j,k)` is the integral of the 3D
Gaussian of each particle over the voxel:

.. math::

    \rho_{\text{sim}}(i,j,k) = \sum_{n=1}^N \int_{V_{ijk}} g(x,y,z;x_n,y_n,z_n)\, dx\, dy\, dz

.. math::

    g = \frac{\epsilon_n}{(2\pi)^{3/2}\sigma_{x,n}\sigma_{y,n}\sigma_{z,n}}
    \exp\!\left( -\frac{1}{2}\left[ \frac{(x-x_n)^2}{\sigma_{x,n}^2}
    + \frac{(y-y_n)^2}{\sigma_{y,n}^2} + \frac{(z-z_n)^2}{\sigma_{z,n}^2} \right] \right)

The integral over a box separates per dimension and is solved with the error
function. For a normal distribution the cumulative distribution function is

.. math::

    \Phi(x; \mu, \sigma) = \frac{1}{2}\left[ 1 + \text{erf}\!\left( \frac{x-\mu}{\sigma\sqrt{2}} \right) \right],
    \qquad \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}\, dt

so the voxel density is a product of CDF differences along each axis:

.. math::

    \rho_{\text{sim}}(i,j,k) = \sum_{n=1}^N \epsilon_n
    \left( \Phi(x_i^{max}) - \Phi(x_i^{min}) \right)
    \left( \Phi(y_j^{max}) - \Phi(y_j^{min}) \right)
    \left( \Phi(z_k^{max}) - \Phi(z_k^{min}) \right)

Derivatives
-----------

The force on atom :math:`n` follows from the chain rule on a variable :math:`v`
(a coordinate or a width):

.. math::

    \frac{\partial V_{Fit}}{\partial v}
    = \frac{\partial V_{Fit}}{\partial \text{c.c.}}\,
      \frac{\partial \text{c.c.}}{\partial \rho_{sim}}\,
      \frac{\partial \rho_{sim}}{\partial v}
    = -k\, \frac{d\,\text{c.c.}}{dv}

since :math:`V_{Fit} = k(1-\text{c.c.})` gives :math:`\partial V_{Fit}/\partial\text{c.c.} = -k`.
The applied force is therefore :math:`-\partial V_{Fit}/\partial v = +k\, (d\,\text{c.c.}/dv)`, 
exactly :math:`k` times the gradient returned by :meth:`~openfit.DensityMap.gradient`. 
The correlation derivative is

.. math::

    \frac{d\,\text{c.c.}}{dv} =
    \frac{\sum_{ijk} \frac{\partial \rho_{\text{sim}}}{\partial v}\, \rho_{\text{exp}}}
         {\sqrt{\sum_{ijk} \rho_{\text{sim}}^2}\, \sqrt{\sum_{ijk} \rho_{\text{exp}}^2}}
    - \frac{\sum_{ijk} \rho_{\text{sim}}\, \frac{\partial \rho_{\text{sim}}}{\partial v}\;
            \sum_{i'j'k'} \rho_{\text{sim}}\, \rho_{\text{exp}}}
           {\left( \sum_{ijk} \rho_{\text{sim}}^2 \right)^{3/2} \sqrt{\sum_{ijk} \rho_{\text{exp}}^2}}

The CDF derivatives are

.. math::

    \frac{\partial \Phi}{\partial \mu} = -\frac{e^{-\frac{(x-\mu)^2}{2\sigma^2}}}{\sqrt{2\pi}\,\sigma},
    \qquad
    \frac{\partial \Phi}{\partial \sigma} = -\frac{(x-\mu)\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}}{\sqrt{2\pi}\,\sigma^2}

so, for example, the derivative of the simulated density with respect to
:math:`x_n` is

.. math::

    \frac{\partial \rho_{\text{sim}}(i,j,k)}{\partial x_n} = \epsilon_n
    \left( \frac{e^{\frac{-(x_i^{min}-x_n)^2}{2\sigma_{x,n}^2}} - e^{\frac{-(x_i^{max}-x_n)^2}{2\sigma_{x,n}^2}}}{\sqrt{2\pi}\,\sigma_{x,n}} \right)
    \left( \Phi(y_j^{max}) - \Phi(y_j^{min}) \right)
    \left( \Phi(z_k^{max}) - \Phi(z_k^{min}) \right)

(the :math:`x_i^{min}` term carries the positive sign because
:math:`\partial \Phi/\partial \mu` is negative). These are exactly what
:meth:`~openfit.DensityMap.gradient` returns (columns ``0:3`` for coordinates,
``3:6`` for widths, ``6`` for the weight).

Choosing the force constant
---------------------------

The bias :math:`V_{Fit} = k\,(1 - \text{c.c.})` is **intensive**: the
cross-correlation is normalized, so :math:`V_{Fit}` is bounded by :math:`\sim k`
no matter how many particles the system has. The force field energy
:math:`V_{ff}`, by contrast, is **extensive** (a sum over bonds, angles, contacts
and non-bonded pairs), so it grows with the particle count :math:`N`. Two
consequences matter when picking ``k``:

* **System size.** The single correlation gradient is shared over all
  :math:`N` particles, so the *per-atom* bias force scales as
  :math:`k\,|\nabla\,\text{c.c.}| \sim k/N`. To keep that per-atom pull — which
  competes with the (intensive) per-atom force-field force — constant across
  systems of different size, scale the constant with the system size. Pass
  ``k_per_particle`` to any builder and OpenFit uses
  :math:`k = k_\text{per particle} \cdot N` instead of the absolute ``k``:

  .. code-block:: python

      Fit.from_amber("model.pdb", "target.mrc", k_per_particle=12.6)  # k = 12.6 * N (kJ/mol)
      Fit.from_amber("model.pdb", "target.mrc", k=6400)               # absolute (default)

  The cryo-EM flexible-fitting literature uses exactly this size scaling: the
  recommended biasing constant for cross-correlation refinement is
  :math:`k = 3\,N_\text{atom}` **kcal/mol** — :math:`\alpha`-helices and
  :math:`\beta`-strands begin to collapse beyond it ([JCIM2025]_, from 29 systems
  at 3.0-6.8 Å). Converting to OpenFit's kJ/mol units that is
  :math:`k_\text{per particle} \approx 12.6`, the recommended *ceiling*.

* **Force-field energy scale.** Size is not the whole story. The *relative*
  weight :math:`V_{Fit}/|V_{ff}|` is set mostly by the force field's intrinsic
  energy scale, which is independent of :math:`N`. Structure-based (SMOG) models
  sit near their energy minimum (tens-hundreds of kJ/mol), so at the default
  ``k`` the bias can equal or exceed :math:`V_{ff}`; stiff all-atom models
  (Amber, CHARMM) carry energies of tens of thousands of kJ/mol, so the same
  bias is only a few percent. Size-normalization does not remove this axis — a
  soft coarse-grained model (e.g. SMOG CA) overfits well below the
  :math:`3\,N_\text{atom}` kcal/mol ceiling and wants a *smaller*
  ``k_per_particle`` than an all-atom model. The most transferable recipe is to
  choose ``k`` so that :math:`V_{Fit}` is a target *fraction* of :math:`|V_{ff}|`,
  which can be calibrated with a single energy evaluation at setup.

.. note::

   Maximizing the correlation alone (as a raw ``k`` scan does) pushes toward
   *over*-fitting: the cross-correlation keeps rising with ``k`` even as the
   model strains. The literature constant [JCIM2025]_ is chosen against a
   **structural** criterion (MolProbity, secondary-structure retention), not
   correlation. On the 4AKE benchmark this is exactly what we see — the soft
   SMOG CA model peaks then degrades once ``k`` exceeds roughly
   :math:`3\,N_\text{atom}` kcal/mol, while the stiffer all-atom models stay
   below that ceiling over the scanned range and keep improving.

   Note that OpenFit's cross-correlation is **centered** (Pearson), whereas the
   reference implementations [cryo_fit]_ use an **uncentered** correlation of the
   same functional form, so :math:`k \approx 12.6\,N_\text{atom}` kJ/mol is a
   calibrated starting ceiling rather than an exact transfer.

References
==========

.. [JCIM2025] "An Empirical Biasing Force Constant to Minimize Overfitting in
   Cryo-EM Flexible Fitting Refinement." *Journal of Chemical Information and
   Modeling* (2025). https://doi.org/10.1021/acs.jcim.5c01424

.. [cryo_fit] "cryo_fit: Democratization of Flexible Fitting for Cryo-EM."
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7112765/ — the MDfit-style
   cross-correlation biasing potential. SMOG/MDfit:
   https://smog-server.org/extension/MDfit.html

.. [OpenSMOG] Oliveira, A. B.; Contessoto, V. G.; et al. "SMOG 2 and OpenSMOG:
   Extending the limits of structure-based models." *Protein Science* **31**,
   158-172 (2022). https://doi.org/10.1002/pro.4209
