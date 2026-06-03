"""High-level flexible-fitting orchestrator.

:class:`Fit` is the one-call entry point: give it an OpenMM system (or build one
with :meth:`Fit.from_amber` / :meth:`Fit.from_smog`) plus a target density, and
it wires up the density-fitting force, drives the simulation with
:meth:`refine`, and saves the result. OpenFit only supplies the density term; the
physical forcefield comes from the OpenMM system you provide.

OpenMM/OpenSMOG are imported lazily so ``import openfit`` stays lightweight.
"""

from pathlib import Path

import numpy as np

from .density import DensityMap
from .forces import DensityForce, DensityForceUpdater

# First letter of a SMOG all-atom atom name is its element.
_ELEMENT_MASS = {"C": 12.0, "N": 14.0, "O": 16.0, "S": 32.0, "H": 1.0, "P": 31.0}


def _as_density(density):
    """Coerce a DensityMap / mrc-path / ndarray into a DensityMap."""
    if isinstance(density, DensityMap):
        return density
    if isinstance(density, (str, Path)):
        return DensityMap.from_mrc(str(density))
    return DensityMap(np.asarray(density))


def _select_platform(name):
    """Return an OpenMM Platform by name, or None to let OpenMM choose."""
    if name is None:
        return None
    import openmm

    return openmm.Platform.getPlatformByName(name)


def _load_structure(path):
    """Parse a ``.pdb`` or ``.cif``/``.pdbx`` structure into (topology, positions)."""
    from openmm import app

    text = str(path)
    if text.lower().endswith((".cif", ".pdbx")):
        structure = app.PDBxFile(text)
    else:
        structure = app.PDBFile(text)
    return structure.topology, structure.positions


_SMOG_FLAGS = {"AA": "-AA", "CA": "-CA", "AAgaussian": "-AAgaussian", "CAgaussian": "-CAgaussian"}


def _generate_smog_model(structure, model, workdir, smog2, smog_adjustpdb):
    """Run SMOG 2 to build an OpenSMOG model from a structure.

    Returns ``(gro, top, xml)`` paths in ``workdir``. ``structure`` must be a
    SMOG-compatible PDB (heavy-atom protein, ATOM records); CIF inputs are first
    converted to PDB. Raises ``RuntimeError`` with the tool output on failure.
    """
    import os
    import shutil
    import subprocess

    if model not in _SMOG_FLAGS:
        raise ValueError(f"unknown SMOG model {model!r}; expected one of {sorted(_SMOG_FLAGS)}")
    adjust_exe = shutil.which(smog_adjustpdb) or (smog_adjustpdb if os.path.exists(smog_adjustpdb) else None)
    smog2_exe = shutil.which(smog2) or (smog2 if os.path.exists(smog2) else None)
    if adjust_exe is None or smog2_exe is None:
        raise RuntimeError(
            "SMOG 2 tools not found. Install SMOG 2 (https://smog-server.org) and put "
            f"'{smog_adjustpdb}'/'{smog2}' on PATH, or pass explicit paths."
        )

    # Absolute path: the SMOG subprocesses run with cwd=workdir.
    structure = os.path.abspath(str(structure))
    # CIF -> PDB (SMOG tools read PDB); plain PDBs are passed through unchanged.
    if structure.lower().endswith((".cif", ".pdbx")):
        from openmm.app import PDBFile

        topology, positions = _load_structure(structure)
        structure = os.path.join(workdir, "input.pdb")
        with open(structure, "w") as handle:
            PDBFile.writeFile(topology, positions, handle)

    adjusted = os.path.join(workdir, "adjusted.pdb")
    base = os.path.join(workdir, "model")
    try:
        subprocess.run(
            [adjust_exe, "-i", structure, "-o", adjusted], cwd=workdir, check=True, capture_output=True, text=True
        )
        subprocess.run(
            [
                smog2_exe,
                "-i",
                adjusted,
                _SMOG_FLAGS[model],
                "-OpenSMOG",
                "-dname",
                base,
                "-OpenSMOGxml",
                base + ".xml",
            ],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"SMOG 2 generation failed:\n{exc.stdout}\n{exc.stderr}") from exc
    return base + ".gro", base + ".top", base + ".xml"


class Fit:
    """Flexibly fit a structure into a density map with an OpenMM simulation.

    The density-fitting force biases the simulation toward higher correlation
    with the target map; a :class:`~openfit.DensityForceUpdater` refreshes it
    during the run. Construct from OpenMM pieces via :meth:`from_system`, or use
    the :meth:`from_amber` / :meth:`from_smog` convenience builders.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        A simulation whose system already contains the density force (built by
        the classmethods; OpenMM forbids adding forces after the Context exists).
    density : DensityMap
        The target density (already configured with coordinates/sigma/epsilon).
    force : DensityForce
        The density force added to the system.
    update_interval : int, optional
        Steps between force refreshes. Default 50.
    k : float, optional
        Force-constant override for the updater.

    Attributes
    ----------
    simulation : openmm.app.Simulation
    density : DensityMap
    force : DensityForce
    """

    def __init__(self, simulation, density, force, *, update_interval=50, k=None):
        self.simulation = simulation
        self.density = density
        self.force = force
        self._updater = DensityForceUpdater(force, interval=update_interval, k=k)
        simulation.reporters.append(self._updater)
        # Prime: sync density coordinates to the current positions and push an
        # initial (non-zero) force so the bias acts from the first step.
        self._sync_coordinates()
        force.update(simulation.context)

    # --- constructors ----------------------------------------------------

    @classmethod
    def from_system(
        cls,
        topology,
        system,
        positions,
        density,
        *,
        integrator=None,
        platform=None,
        k=3200,
        update_interval=50,
        backend="python",
        rigid_search=False,
    ):
        """Build a :class:`Fit` from OpenMM pieces (bring-your-own forcefield).

        Adds the density force to ``system`` (before the Context is created),
        sets the periodic box from the map, builds the ``Simulation`` and sets
        ``positions``.

        Parameters
        ----------
        topology : openmm.app.Topology
        system : openmm.System
            Your forcefield system; the density force is added to it.
        positions : array-like with units
            Initial positions (e.g. ``openmm`` Quantity in nm).
        density : DensityMap or str or numpy.ndarray
            Target density (or an MRC path / array). Should already have its
            coordinates/sigma/epsilon set to match the system's particles.
        integrator : openmm.Integrator, optional
            Defaults to a 300 K LangevinMiddle integrator.
        platform : str, optional
            OpenMM platform name (e.g. ``"CUDA"``); default lets OpenMM choose.
        k : float, optional
            Density force constant.
        update_interval : int, optional
            Steps between force refreshes.
        backend : {"python", "native"}, optional
            ``"python"`` (default) computes the gradient in Python and injects it
            via a tabulated force. ``"native"`` (a C++/CUDA OpenMM plugin) is
            planned for the performance phase and raises ``NotImplementedError``.
        rigid_search : bool or dict, optional
            Optional rigid-body pre-placement. Default ``False`` — the structure
            is assumed to be **already aligned** to the map. Pass ``True`` to run
            :meth:`dock` (a coarse orientation/translation scan) first, or a dict
            of :meth:`~openfit.DensityMap.rigid_fit` keyword arguments
            (e.g. ``{"n_rotations": 300}``).
        """
        import openmm
        from openmm import app, unit

        density = _as_density(density)
        system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in density.periodic_vectors()])

        if backend == "python":
            force = DensityForce(density, k=k)
            force.add_to(system)
        elif backend == "native":
            raise NotImplementedError(
                "backend='native' (a C++/CUDA OpenMM force plugin) is planned for the "
                "performance phase; use backend='python' for now."
            )
        else:
            raise ValueError(f"unknown backend {backend!r}; expected 'python' or 'native'.")

        if integrator is None:
            integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)

        platform_obj = _select_platform(platform)
        if platform_obj is None:
            simulation = app.Simulation(topology, system, integrator)
        else:
            simulation = app.Simulation(topology, system, integrator, platform_obj)
        simulation.context.setPositions(positions)

        fit = cls(simulation, density, force, update_interval=update_interval, k=k)
        if rigid_search:
            fit.dock(**(rigid_search if isinstance(rigid_search, dict) else {}))
        return fit

    @classmethod
    def from_amber(
        cls,
        pdb,
        density,
        *,
        forcefield=("amber14-all.xml", "implicit/gbn2.xml"),
        sigma=2.0,
        add_hydrogens=True,
        **kwargs,
    ):
        """Build a :class:`Fit` for an all-atom MM model from a PDB (experimental).

        Builds the system with ``openmm.app.ForceField`` (Amber14 + implicit
        solvent by default), assuming a clean PDB. Per-atom weights default to
        the atomic masses. Remaining keyword arguments are passed to
        :meth:`from_system`.
        """
        from openmm import app, unit

        pdbfile = app.PDBFile(str(pdb))
        forcefield_obj = app.ForceField(*forcefield)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
        if add_hydrogens:
            modeller.addHydrogens(forcefield_obj)
        system = forcefield_obj.createSystem(
            modeller.topology, nonbondedMethod=app.CutoffNonPeriodic, constraints=app.HBonds
        )

        density = _as_density(density)
        coords = np.array(modeller.positions.value_in_unit(unit.angstrom))
        masses = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
        )
        density.set_coordinates(coords, np.full((len(coords), 3), sigma), masses)

        return cls.from_system(modeller.topology, system, modeller.positions, density, **kwargs)

    @classmethod
    def from_charmm(
        cls,
        structure,
        density,
        *,
        forcefield=("charmm36.xml", "charmm36/water.xml"),
        sigma=2.0,
        add_hydrogens=True,
        **kwargs,
    ):
        """Build a :class:`Fit` for a CHARMM36 all-atom model from a PDB/CIF (experimental).

        Uses OpenMM's bundled CHARMM36 force field via ``openmm.app.ForceField``,
        assuming a clean structure. Per-atom weights default to the atomic
        masses. Remaining keyword arguments are passed to :meth:`from_system`.

        Parameters
        ----------
        structure : str or path-like
            Input ``.pdb`` or ``.cif`` structure.
        density : DensityMap or str or numpy.ndarray
            Target density (or MRC path / array).
        forcefield : tuple of str, optional
            ForceField XML files (default CHARMM36 protein + water).
        sigma : float, optional
            Gaussian width in Angstrom (default 2.0).
        add_hydrogens : bool, optional
            Add missing hydrogens with ``Modeller`` (default True).
        """
        from openmm import app, unit

        topology, positions = _load_structure(structure)
        forcefield_obj = app.ForceField(*forcefield)
        modeller = app.Modeller(topology, positions)
        if add_hydrogens:
            modeller.addHydrogens(forcefield_obj)
        system = forcefield_obj.createSystem(
            modeller.topology, nonbondedMethod=app.CutoffNonPeriodic, constraints=app.HBonds
        )

        density = _as_density(density)
        coords = np.array(modeller.positions.value_in_unit(unit.angstrom))
        masses = np.array(
            [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
        )
        density.set_coordinates(coords, np.full((len(coords), 3), sigma), masses)

        return cls.from_system(modeller.topology, system, modeller.positions, density, **kwargs)

    @classmethod
    def from_smog(
        cls,
        gro,
        top,
        xml,
        density,
        *,
        sigma=1.5,
        epsilon="mass",
        temperature=0.5,
        time_step=0.002,
        collision_rate=1.0,
        r_cutoff=1.2,
        platform="CPU",
        name="openfit",
        k=3200,
        update_interval=50,
        rigid_search=False,
    ):
        """Build a :class:`Fit` from an OpenSMOG structure-based model.

        Ports the validated flexible-fitting workflow: load the SMOG model, add
        the density force, strip the COM-motion remover (so the force can
        translate the molecule), and let OpenSMOG build the simulation.

        Parameters
        ----------
        gro, top, xml : str or path-like
            SMOG 2 model files.
        density : DensityMap or str or numpy.ndarray
            Target density (or MRC path / array).
        sigma : float or array-like, optional
            Gaussian width (scalar broadcast to ``(n, 3)``).
        epsilon : {"mass"}, None or array-like, optional
            Per-atom weight; ``"mass"`` (default) uses atomic masses from the
            atom names.
        platform : str, optional
            OpenSMOG platform (default ``"CPU"``).
        rigid_search : bool or dict, optional
            Optional rigid-body pre-placement. Default ``False`` — the structure
            is assumed already aligned to the map. ``True`` runs :meth:`dock`
            first; a dict passes keyword arguments to
            :meth:`~openfit.DensityMap.rigid_fit`.
        """
        import openmm
        from OpenSMOG import SBM

        sbm = SBM(
            name=name,
            time_step=time_step,
            collision_rate=collision_rate,
            r_cutoff=r_cutoff,
            temperature=temperature,
            warn=False,
        )
        sbm.setup_openmm(platform=platform, GPUindex="default")
        import tempfile

        sbm.saveFolder(tempfile.mkdtemp(prefix="openfit_smog_"))
        sbm.loadSystem(Grofile=str(gro), Topfile=str(top), Xmlfile=str(xml))

        density = _as_density(density)
        sbm.system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in density.periodic_vectors()])
        force = DensityForce(density, k=k)
        force.add_to(sbm.system)
        for i, f in reversed(list(enumerate(sbm.system.getForces()))):
            if isinstance(f, openmm.CMMotionRemover):
                sbm.system.removeForce(i)

        sbm.loaded = False
        sbm.createSimulation()

        names = sbm.Gro.atomNames
        if isinstance(epsilon, str) and epsilon == "mass":
            weights = np.array([_ELEMENT_MASS[a[0]] for a in names], dtype=float)
        elif epsilon is None:
            weights = np.ones(len(names))
        else:
            weights = np.asarray(epsilon, dtype=float)
        widths = np.full((len(names), 3), sigma) if np.isscalar(sigma) else np.asarray(sigma)

        state = sbm.simulation.context.getState(getPositions=True)
        coords = np.array(state.getPositions().value_in_unit(openmm.unit.angstrom))
        density.set_coordinates(coords, widths, weights)

        fit = cls(sbm.simulation, density, force, update_interval=update_interval, k=k)
        if rigid_search:
            fit.dock(**(rigid_search if isinstance(rigid_search, dict) else {}))
        return fit

    @classmethod
    def from_smog_structure(
        cls,
        structure,
        density,
        *,
        model="AA",
        smog2="smog2",
        smog_adjustpdb="smog_adjustPDB",
        **kwargs,
    ):
        """Build a :class:`Fit` by generating a SMOG model from a structure.

        Runs SMOG 2 (``smog_adjustPDB`` + ``smog2 ... -OpenSMOG``) on ``structure``
        to produce the ``.gro``/``.top``/``.xml`` model, then delegates to
        :meth:`from_smog`. SMOG 2 must be installed (external tool); the input
        must be a SMOG-compatible heavy-atom protein PDB/CIF.

        Parameters
        ----------
        structure : str or path-like
            Input ``.pdb`` or ``.cif`` (a cleaned, single-conformer protein).
        density : DensityMap or str or numpy.ndarray
            Target density (or MRC path / array).
        model : {"AA", "CA", "AAgaussian", "CAgaussian"}, optional
            SMOG model type (default all-atom).
        smog2, smog_adjustpdb : str, optional
            Names/paths of the SMOG 2 executables (resolved from PATH).
        **kwargs
            Passed to :meth:`from_smog` (e.g. ``k``, ``update_interval``,
            ``platform``, ``rigid_search``).
        """
        import tempfile

        workdir = tempfile.mkdtemp(prefix="openfit_smog_")
        gro, top, xml = _generate_smog_model(structure, model, workdir, smog2, smog_adjustpdb)
        return cls.from_smog(gro, top, xml, density, **kwargs)

    @classmethod
    def from_awsem(cls, structure, density, *, chains="A", k_awsem=1.0, sigma=3.0, force_setup=None, **kwargs):
        """Build a :class:`Fit` for an OpenAWSEM coarse-grained model (experimental).

        Builds a Cα/Cβ/O coarse-grained AWSEM system from a PDB/CIF via
        `OpenAWSEM <https://github.com/npschafer/openawsem>`_. By default a
        minimal, auxiliary-file-free force set is used (backbone + contact terms;
        the fragment-memory, Debye-Hückel and secondary-structure-weight terms,
        which need generated input files, are omitted). Requires ``openawsem``.

        Parameters
        ----------
        structure : str or path-like
            Input ``.pdb`` or ``.cif``.
        density : DensityMap or str or numpy.ndarray
            Target density (or MRC path / array).
        chains : str, optional
            Chain id(s) to simulate (default ``"A"``).
        k_awsem : float, optional
            Overall AWSEM energy scale (default 1.0).
        sigma : float, optional
            Gaussian width for the CG beads in Angstrom (default 3.0).
        force_setup : callable, optional
            ``oa -> list`` returning the AWSEM force terms for an
            ``OpenMMAWSEMSystem``; overrides the default minimal set (e.g. pass
            ``openawsem.scripts.forces_setup.set_up_forces`` with its input files).
        **kwargs
            Passed to :meth:`from_system`.
        """
        import os
        import shutil
        import tempfile

        import numpy as np
        from openmm import unit

        import openawsem
        from openawsem import OpenMMAWSEMSystem, prepare_pdb
        from openawsem.functionTerms import basicTerms, contactTerms

        structure = os.path.abspath(str(structure))
        workdir = tempfile.mkdtemp(prefix="openfit_awsem_")
        cwd = os.getcwd()
        try:
            os.chdir(workdir)
            local_pdb = "structure.pdb"
            if structure.lower().endswith((".cif", ".pdbx")):
                from openmm.app import PDBFile

                topology, positions = _load_structure(structure)
                with open(local_pdb, "w") as handle:
                    PDBFile.writeFile(topology, positions, handle)
            else:
                shutil.copy(structure, local_pdb)

            prepare_pdb(local_pdb, chains)
            oa = OpenMMAWSEMSystem(
                "structure-openmmawsem.pdb", chains=chains, k_awsem=k_awsem, xml_filename=openawsem.xml
            )
            if force_setup is None:
                forces = [
                    basicTerms.con_term(oa),
                    basicTerms.chain_term(oa),
                    basicTerms.chi_term(oa),
                    basicTerms.excl_term(oa, periodic=False),
                    basicTerms.rama_term(oa),
                    basicTerms.rama_proline_term(oa),
                    contactTerms.contact_term(oa),
                ]
            else:
                forces = force_setup(oa)
            oa.addForcesWithDefaultForceGroup(forces)
            topology, system, positions = oa.pdb.topology, oa.system, oa.pdb.positions
        finally:
            os.chdir(cwd)

        density = _as_density(density)
        coords = np.array(positions.value_in_unit(unit.angstrom))
        density.set_coordinates(coords, np.full((len(coords), 3), sigma), np.ones(len(coords)))
        return cls.from_system(topology, system, positions, density, **kwargs)

    # --- driving ---------------------------------------------------------

    def _sync_coordinates(self):
        """Update the density's coordinates from the current simulation state."""
        import openmm

        state = self.simulation.context.getState(getPositions=True)
        coords = np.array(state.getPositions().value_in_unit(openmm.unit.angstrom))
        self.density.set_coordinates(coords)

    def dock(self, **rigid_kwargs):
        """Rigid-body place the structure into the map before flexible fitting.

        Runs :meth:`~openfit.DensityMap.rigid_fit` (an orientation + translation
        scan) on the current structure and writes the best pose back into the
        simulation. Keyword arguments are forwarded to ``rigid_fit`` (e.g.
        ``n_rotations``, ``n_translations``). Assumes the structure fits inside
        the map box. Returns the ``rigid_fit`` result dict.
        """
        import openmm

        self._sync_coordinates()
        result = self.density.rigid_fit(**rigid_kwargs)
        # internal frame -> world (Angstrom) -> nm
        world = self.density.coordinates + self.density.origin - self.density.voxel_size / 2
        self.simulation.context.setPositions((world / 10.0) * openmm.unit.nanometer)
        self.force.update(self.simulation.context)
        return result

    @property
    def cc(self):
        """Current map correlation coefficient (recomputed from live positions)."""
        self._sync_coordinates()
        return self.density.correlation()

    def minimize(self, max_iterations=0):
        """Run OpenMM energy minimization. Returns the resulting correlation."""
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
        return self.cc

    def refine(self, steps, minimize=False, record_interval=None, trajectory=None, trajectory_interval=1000):
        """Run the flexible fit, returning the correlation history.

        Parameters
        ----------
        steps : int
            Total MD steps to run (the force is refreshed every
            ``update_interval`` steps by the attached updater).
        minimize : bool, optional
            Energy-minimize before the run. Default False.
        record_interval : int, optional
            Record the correlation every this many steps (default: once at the
            end).
        trajectory : str or path-like, optional
            If given, write the trajectory to this ``.dcd`` file during the run.
        trajectory_interval : int, optional
            Steps between trajectory frames (default 1000).

        Returns
        -------
        dict
            ``{"correlation", "steps", "history"}``.
        """
        if minimize:
            self.minimize()

        reporter = None
        if trajectory is not None:
            from openmm import app

            reporter = app.DCDReporter(str(trajectory), trajectory_interval)
            self.simulation.reporters.append(reporter)

        record_interval = record_interval or steps
        history = []
        done = 0
        try:
            while done < steps:
                chunk = min(record_interval, steps - done)
                self.simulation.step(chunk)
                done += chunk
                history.append(self.cc)
        finally:
            if reporter is not None:
                self.simulation.reporters.remove(reporter)
                reporter._out.close()
        return {
            "correlation": history[-1] if history else self.cc,
            "steps": steps,
            "history": np.asarray(history),
        }

    # alias for OpenMM users
    run = refine

    # --- output ----------------------------------------------------------

    def save(self, pdb_path):
        """Write the current structure to a PDB file."""
        from openmm import app

        state = self.simulation.context.getState(getPositions=True)
        with open(pdb_path, "w") as handle:
            app.PDBFile.writeFile(self.simulation.topology, state.getPositions(), handle)

    def save_map(self, mrc_path, **kwargs):
        """Write the current simulated density to an MRC file."""
        self._sync_coordinates()
        self.density.save_mrc(str(mrc_path), **kwargs)

    def __repr__(self):
        n = self.simulation.system.getNumParticles()
        return f"Fit(particles={n}, density={self.density!r})"
