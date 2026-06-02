import logging
from pathlib import Path

import numpy as np
from scipy.special import erf

from ._kernels import dcorr_v3, sim_map


class DensityMap:
    """Fit a set of particles to a 3D density map using anisotropic Gaussians.

    Each particle is represented by an anisotropic 3D Gaussian. The class
    computes the simulated density on the experimental grid, its correlation
    with the experimental map, and the analytical gradient of that correlation
    with respect to the particle coordinates and Gaussian widths. The gradient
    can drive a direct optimization (:meth:`fit`) or be injected as a force into
    an OpenMM simulation (:meth:`add_force` / :meth:`update_force`).

    Parameters
    ----------
    experimental_map : numpy.ndarray
        The experimental density, a 3D array in ``(z, y, x)`` order. Stored
        normalized to zero mean and unit standard deviation.
    voxel_size : array-like of float, optional
        Edge length of a voxel along ``(x, y, z)`` in Angstrom. Defaults to
        ``(1, 1, 1)``.
    origin : array-like of float, optional
        World coordinate of the first voxel's corner. Defaults to ``(0, 0, 0)``.
    dtype : numpy dtype, optional
        Floating-point precision used internally. Defaults to ``numpy.float64``.

    Attributes
    ----------
    coordinates : numpy.ndarray
        Particle coordinates in the internal grid frame, shape ``(n, 3)``.
    sigma : numpy.ndarray
        Per-particle anisotropic Gaussian widths, shape ``(n, 3)``.
    epsilon : numpy.ndarray
        Per-particle weights (e.g. atomic mass), shape ``(n,)``.
    """

    def __init__(self, experimental_map, voxel_size=None, origin=None, dtype=np.float64):
        self.dtype = dtype

        # Convert inputs to numpy arrays with specified dtype, handling defaults
        self.experimental_map = np.asarray(experimental_map, dtype=self.dtype)
        if voxel_size is None:
            self.voxel_size = np.array([1, 1, 1], dtype=self.dtype)
            logging.debug("Voxel size not provided. Assuming 1 Angstrom for all dimensions.")
        elif type(voxel_size) is np.recarray:
            self.voxel_size = np.array([voxel_size["x"], voxel_size["y"], voxel_size["z"]], dtype=float)
        else:
            self.voxel_size = np.asarray(voxel_size, dtype=self.dtype)

        if origin is None:
            self.origin = np.array([0, 0, 0], dtype=self.dtype)
            logging.debug("Origin not provided. Assuming (0, 0, 0).")
        elif type(origin) is np.recarray:
            self.origin = np.array([origin["x"], origin["y"], origin["z"]], dtype=float)
        else:
            self.origin = np.asarray(origin, dtype=self.dtype)

        # Validate the dimensions of inputs
        if self.voxel_size.ndim != 1 or self.voxel_size.size != 3:
            raise ValueError("voxel_size must be a one-dimensional numpy array of size 3.")
        if self.experimental_map.ndim != 3:
            raise ValueError("experimental_map must be a 3D numpy array.")

        self.n_voxels = np.array(self.experimental_map.shape)[::-1]
        self.padding = None
        self.voxel_limits = None
        self.coordinates = None
        self.sigma = None
        self.epsilon = None
        self.force = None

    @property
    def experimental_map(self):
        return self._experimental_map

    @experimental_map.setter
    def experimental_map(self, value):
        """Store the experimental map normalized with zero mean and zero unit standard deviation."""
        self._experimental_map = value
        self.experimental_map_mean = self._experimental_map.mean()
        self._experimental_map -= self.experimental_map.mean()
        self.experimental_map_std = self._experimental_map.std()
        if self.experimental_map_std > 0:
            self._experimental_map /= self.experimental_map_std
        else:
            logging.warning("The experimental map is uniform. The standard deviation is zero.")

    def __repr__(self):
        n = 0 if self.coordinates is None else len(self.coordinates)
        return (
            f"DensityMap(n_voxels={tuple(int(x) for x in self.n_voxels)}, "
            f"voxel_size={tuple(float(x) for x in self.voxel_size)}, particles={n})"
        )

    @classmethod
    def from_mrc(cls, mrc_file, cutoff_min=None, cutoff_max=None, dtype=np.float64):
        """Create a :class:`DensityMap` from an MRC/CCP4 density file.

        Voxel size and origin are read from the file header.

        Parameters
        ----------
        mrc_file : str or path-like
            Path to the MRC/CCP4 map.
        cutoff_min, cutoff_max : float, optional
            If given, clamp densities below ``cutoff_min`` to 0 and above
            ``cutoff_max`` to ``cutoff_max`` before normalization.
        dtype : numpy dtype, optional
            Internal precision.

        Returns
        -------
        DensityMap
        """
        import mrcfile

        with mrcfile.open(mrc_file) as mrc:
            data = mrc.data
            voxel_size = mrc.voxel_size
            header = mrc.header

        if cutoff_min:
            data[data < cutoff_min] = 0
        if cutoff_max:
            data[data > cutoff_max] = cutoff_max
        return cls(
            experimental_map=data.transpose(header["mapc"] - 1, header["mapr"] - 1, header["maps"] - 1),
            voxel_size=np.array([voxel_size["x"], voxel_size["y"], voxel_size["z"]]),
            origin=np.array([header["origin"]["x"], header["origin"]["y"], header["origin"]["z"]]),
            dtype=dtype,
        )

    @classmethod
    def from_dimensions(cls, min_coords, max_coords, voxel_size, dtype=np.float64):
        """Create an empty-grid :class:`DensityMap` covering a coordinate box.

        Useful when there is no experimental map yet and you only need a grid to
        rasterize a structure onto (e.g. to generate a synthetic density).

        Parameters
        ----------
        min_coords, max_coords : array-like of float
            Lower and upper ``(x, y, z)`` bounds, in Angstrom.
        voxel_size : array-like of float
            Voxel edge lengths ``(x, y, z)``.
        dtype : numpy dtype, optional
            Internal precision.

        Returns
        -------
        DensityMap
        """
        min_coords = np.asarray(min_coords, dtype=dtype)
        max_coords = np.asarray(max_coords, dtype=dtype)
        voxel_size = np.asarray(voxel_size, dtype=dtype)

        n_voxels = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
        origin = (min_coords + max_coords) / 2 - voxel_size * n_voxels / 2 + voxel_size / 2
        return cls(np.empty(n_voxels[::-1], dtype=dtype), voxel_size=voxel_size, origin=origin)

    @classmethod
    def from_scene(cls, scene, voxel_size=(2.0, 2.0, 2.0), padding=10.0, sigma=2.0, epsilon="mass", dtype=np.float64):
        """Create a :class:`DensityMap` from a `MolScene <https://github.com/cabb99/molscene>`_ ``Scene``.

        Builds an empty grid sized to enclose the structure (plus ``padding``),
        and sets the particle coordinates, widths and weights from the scene.
        The particle weights default to the atomic masses.

        Parameters
        ----------
        scene : molscene.Scene
            A parsed structure (e.g. ``molscene.Scene.from_pdb(...)``).
        voxel_size : array-like of float, optional
            Voxel edge lengths ``(x, y, z)``. Defaults to ``(2, 2, 2)``.
        padding : float, optional
            Angstrom of empty space added around the structure's bounding box.
        sigma : float or array-like, optional
            Gaussian width; a scalar is broadcast to all particles as ``(n, 3)``.
        epsilon : {"mass"}, None or array-like, optional
            Per-particle weight. ``"mass"`` (default) uses ``scene.compute_mass()``;
            ``None`` uses ones; otherwise an explicit ``(n,)`` array.
        dtype : numpy dtype, optional
            Internal precision.

        Returns
        -------
        DensityMap
        """
        coords = np.asarray(scene.get_coordinates().to_numpy(), dtype=dtype)
        if isinstance(epsilon, str) and epsilon == "mass":
            weights = np.asarray(scene.compute_mass()["mass"].to_numpy(), dtype=dtype)
        elif epsilon is None:
            weights = np.ones(len(coords), dtype=dtype)
        else:
            weights = np.asarray(epsilon, dtype=dtype)

        voxel_size = np.asarray(voxel_size, dtype=dtype)
        fit = cls.from_dimensions(coords.min(0) - padding, coords.max(0) + padding, voxel_size, dtype=dtype)

        if np.isscalar(sigma):
            widths = np.full((len(coords), 3), sigma, dtype=dtype)
        else:
            widths = np.asarray(sigma, dtype=dtype)
        fit.set_coordinates(coords, sigma=widths, epsilon=weights)
        return fit

    def save_mrc(self, mrc_file, experimental=False, rescale=True):
        """Write a density map to an MRC file.

        Parameters
        ----------
        mrc_file : str or path-like
            Output path (overwritten if it exists).
        experimental : bool, optional
            If True, write the (denormalized) experimental map; otherwise write
            the simulated map. Default False.
        rescale : bool, optional
            For the simulated map, rescale to its own mean/std before applying
            the experimental scale (True) or apply the experimental scale
            directly (False).
        """
        import mrcfile
        import datetime

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        data_description = [f"Data generated on {current_date}", "Simulated density map."]

        if experimental:
            # Copy so that the in-place denormalization below does not corrupt
            # the stored (normalized) experimental map.
            map_data = self.experimental_map.copy()
        else:
            map_data = self.simulation_map()
            if rescale:
                map_data -= map_data.mean()
                map_data /= map_data.std()
            else:
                map_data -= self.experimental_map_mean
                map_data /= self.experimental_map_std

        # Revert the map normalization. It will revert the simulation map too with the experimental map scale
        map_data *= self.experimental_map_std
        map_data += self.experimental_map_mean

        # Assuming mrc is your MRC file object after opening it with mrcfile
        with mrcfile.new(mrc_file, overwrite=True) as mrc:
            mrc.set_data(map_data.astype(np.float32))
            mrc.voxel_size = tuple(self.voxel_size)

            # Setting header values directly can be error-prone due to potential key
            # mismatches or incorrect assignments; instead, use the attributes provided
            # by the mrcfile library when possible.
            mrc.header.mapc, mrc.header.mapr, mrc.header.maps = 1, 2, 3
            mrc.header.mode = 2  # Mode 2 is commonly used for 32-bit floating point data
            mrc.header.mz, mrc.header.my, mrc.header.mx = map_data.shape
            mrc.header.nz, mrc.header.ny, mrc.header.nx = map_data.shape
            mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 0, 0, 0
            mrc.header.cellb = (90.0, 90.0, 90.0)

            # Calculate cell dimensions based on voxel size and shape
            mrc.header.cella = (
                map_data.shape[2] * self.voxel_size[2],
                map_data.shape[1] * self.voxel_size[1],
                map_data.shape[0] * self.voxel_size[0],
            )

            # Origin
            mrc.header.origin = tuple(self.origin)
            mrc.header.nlabl = len(data_description)
            for i in range(mrc.header.nlabl):
                mrc.header.label[i] = data_description[i].encode("ascii")[:80]

            mrc.update_header_stats()
            mrc.flush()

    def compute_forces(self, topology_file, trajectory_file, output_file=None, overwrite=False):
        import mdtraj

        topology = mdtraj.load(topology_file)
        trajectory = mdtraj.load(trajectory_file, top=topology)

        if output_file is None:
            output_file = trajectory_file.replace(".dcd", "_forces.txt")
        if Path(output_file).exists() and not overwrite:
            raise FileExistsError(f"{output_file} already exists.")

        # Saving the forces to a file
        with open(output_file, "w") as file:
            for frame_idx, frame in enumerate(trajectory):
                self.set_coordinates(frame.xyz[0] * 10, self.sigma, self.epsilon)
                frame_forces = self.gradient()
                file.write(f"Frame {frame_idx}\n")
                for atom_idx, force in enumerate(frame_forces):
                    file.write(
                        f"{atom_idx} {force[0]} {force[1]} {force[2]} {force[3]} {force[4]} {force[5]} {force[6]}\n"
                    )

    def add_force(self, system):
        """Add the density-fitting force to an OpenMM ``System``.

        Registers a ``CustomCompoundBondForce`` whose per-particle force is read
        from a tabulated function; refresh it during the simulation with
        :meth:`update_force`. The force is stored on ``self.force``.

        Parameters
        ----------
        system : openmm.System
            The system to add the force to. Must already contain the particles.

        Returns
        -------
        openmm.CustomCompoundBondForce
            The force that was added.
        """
        import openmm

        n_particles = system.getNumParticles()

        force = openmm.CustomCompoundBondForce(1, "-k(i,0)*x1-k(i,1)*y1-k(i,2)*z1")
        force_array = np.zeros((n_particles, 3))
        force_vectors = openmm.Discrete2DFunction(force_array.shape[0], force_array.shape[1], force_array.T.flatten())
        force.addTabulatedFunction("k", force_vectors)
        force.addPerBondParameter("i")
        for i in range(n_particles):
            force.addBond([i], [i])
        force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(force)
        self.force = force
        return force

    def periodic_vectors(self):
        """Periodic box vectors matching the map extent, in nanometres.

        Returns
        -------
        tuple of list of float
            The three box vectors suitable for
            ``System.setDefaultPeriodicBoxVectors``.
        """
        return (
            [self.voxel_size[0] * self.n_voxels[0] / 10, 0, 0],
            [0, self.voxel_size[1] * self.n_voxels[1] / 10, 0],
            [0, 0, self.voxel_size[2] * self.n_voxels[2] / 10],
        )

    def update_coordinates(self, simulation):
        import openmm

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        coordinates = np.array(positions.value_in_unit(openmm.unit.angstrom))
        self.set_coordinates(coordinates)

    def update_force(self, simulation, update_coordinates=True, k=3200, force=None, force_array=None):
        """Refresh the fitting force from the current simulation state.

        Reads the current positions (unless ``update_coordinates`` is False),
        recomputes the correlation gradient scaled by ``k``, writes it into the
        tabulated function, and pushes it to the OpenMM ``Context`` without a
        rebuild.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The running simulation (only ``simulation.context`` is used).
        update_coordinates : bool, optional
            Re-read positions from the context before computing the gradient.
        k : float, optional
            Force constant scaling the correlation gradient.
        force : openmm.Force, optional
            Force to update; defaults to the one created by :meth:`add_force`.
        force_array : numpy.ndarray, optional
            Explicit ``(n, 3)`` force values to use instead of the computed gradient.

        Returns
        -------
        numpy.ndarray
            The ``(n, 3)`` force array written to the context.
        """
        if update_coordinates:
            self.update_coordinates(simulation)
        if force is None:
            force = self.force
        if force_array is None:
            force_array = k * self.gradient()[:, :3]
        tabulated_function = self.force.getTabulatedFunction(0)
        params = tabulated_function.getFunctionParameters()
        params[2] = force_array.T.ravel()
        tabulated_function.setFunctionParameters(*params)
        force.updateParametersInContext(simulation.context)
        return force_array

    def set_coordinates(self, coordinates, sigma=None, epsilon=None):
        """Set the particle coordinates and (optionally) widths and weights.

        ``sigma`` and ``epsilon`` are remembered between calls, so passing only
        ``coordinates`` reuses the previously set values.

        Parameters
        ----------
        coordinates : array-like, shape (n, 3)
            Particle coordinates in Angstrom (world frame; shifted internally).
        sigma : array-like, shape (n, 3), optional
            Anisotropic Gaussian widths. Defaults to ones on first call.
        epsilon : array-like, shape (n,), optional
            Per-particle weights. Defaults to ones on first call.

        Raises
        ------
        ValueError
            If any of the input shapes are inconsistent.
        """
        coordinates = np.asarray(coordinates, dtype=self.dtype)
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=self.dtype)
        if epsilon is not None:
            epsilon = np.asarray(epsilon, dtype=self.dtype)

        # Validate shapes
        if coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (n, 3)")

        if sigma is not None and (sigma.shape[0] != coordinates.shape[0] or sigma.shape[1] != 3):
            raise ValueError("sigma must have shape (n, 3)")
        elif sigma is None and self.sigma is None:
            sigma = np.ones((coordinates.shape[0], 3))
        elif sigma is None and self.sigma is not None:
            sigma = self.sigma

        if epsilon is not None and epsilon.shape != (coordinates.shape[0],):
            raise ValueError("epsilon must have shape (n,)")
        elif epsilon is None and self.epsilon is None:
            epsilon = np.ones(coordinates.shape[0])
        elif epsilon is None and self.epsilon is not None:
            epsilon = self.epsilon

        self.coordinates = (
            coordinates - self.origin + self.voxel_size / 2
        )  # Origin of the coordinates is in the center of the first voxel
        self.sigma = sigma
        self.epsilon = epsilon
        self.setup_map(sigma)
        self.fix_bounds()

    def setup_map(self, sigma):
        # Assumes sigma has been validated and is available
        self.padding = int(np.ceil(5 * np.max(sigma.max(axis=0) / self.voxel_size)))
        self.voxel_limits = [
            np.arange(-self.padding, self.n_voxels[i] + self.padding + 1) * self.voxel_size[i] for i in range(3)
        ]

    def fix_bounds(self):
        # Fix the coordinates to be within the bounds of the experimental map
        max_bounds = self.n_voxels * self.voxel_size
        self.coordinates = np.mod(self.coordinates, max_bounds)

    def fold_padding(self, volume_map):
        p = self.padding
        vp = volume_map.copy()
        if p > 0 and len(volume_map.shape) == 3:
            vp[-2 * p : -p, :, :] += vp[:p, :, :]
            vp[:, -2 * p : -p, :] += vp[:, :p, :]
            vp[:, :, -2 * p : -p] += vp[:, :, :p]
            vp[p : 2 * p, :, :] += vp[-p:, :, :]
            vp[:, p : 2 * p, :] += vp[:, -p:, :]
            vp[:, :, p : 2 * p] += vp[:, :, -p:]
            vp = vp[p:-p, p:-p, p:-p]
        elif p > 0 and len(volume_map.shape) == 4:
            vp[:, -2 * p : -p, :, :] += vp[:, :p, :, :]
            vp[:, :, -2 * p : -p, :] += vp[:, :, :p, :]
            vp[:, :, :, -2 * p : -p] += vp[:, :, :, :p]
            vp[:, p : 2 * p, :, :] += vp[:, -p:, :, :]
            vp[:, :, p : 2 * p, :] += vp[:, :, -p:, :]
            vp[:, :, :, p : 2 * p] += vp[:, :, :, -p:]
            vp = vp[:, p:-p, p:-p, p:-p]
        return vp

    def simulation_map(self, normalize=False, device="cpu"):
        """Compute the simulated density on the experimental grid.

        Parameters
        ----------
        normalize : bool, optional
            If True, return the map normalized to zero mean and unit standard
            deviation.
        device : {"cpu"}, optional
            Compute backend. Only the CPU Numba kernel is available today; a
            ``"cuda"`` kernel arrives in the performance phase.

        Returns
        -------
        numpy.ndarray
            The simulated density, same shape as ``experimental_map``.
        """
        if device != "cpu":
            raise NotImplementedError(
                f"device={device!r} is not available yet; only 'cpu' is supported "
                "(GPU kernels arrive in the performance phase)."
            )
        sim = sim_map(self.coordinates, self.n_voxels, self.voxel_size, self.sigma, self.epsilon, self.padding, 5)
        if normalize:
            sim_map_mean = sim.mean()
            sim_map_std = sim.std()
            return (sim - sim_map_mean) / sim_map_std
        else:
            return sim

    def simulation_map_numpy(self):
        sigma = self.sigma * np.sqrt(2)
        phix = (1 + erf((self.voxel_limits[0] - self.coordinates[:, None, 0]) / sigma[:, None, 0])) / 2
        phiy = (1 + erf((self.voxel_limits[1] - self.coordinates[:, None, 1]) / sigma[:, None, 1])) / 2
        phiz = (1 + erf((self.voxel_limits[2] - self.coordinates[:, None, 2]) / sigma[:, None, 2])) / 2

        dphix = phix[:, 1:] - phix[:, :-1]
        dphiy = phiy[:, 1:] - phiy[:, :-1]
        dphiz = phiz[:, 1:] - phiz[:, :-1]

        smap = (
            self.epsilon[:, None, None, None]
            * dphix[:, None, None, :]
            * dphiy[:, None, :, None]
            * dphiz[:, :, None, None]
        ).sum(axis=0)

        return self.fold_padding(smap)

    def correlation(self):
        """Cross-correlation between the simulated and experimental densities.

        Returns
        -------
        float
            The correlation coefficient in ``[-1, 1]`` (1 is a perfect match).
        """
        simulation_map = self.simulation_map()
        sim_map_mean = simulation_map.mean()
        sim_map_std = simulation_map.std()
        return ((simulation_map - sim_map_mean) / sim_map_std * (self.experimental_map)).mean()

    def gradient_numerical(self, delta=1e-5):
        num_derivatives = np.zeros((self.coordinates.shape[0], 7))

        for i in range(self.coordinates.shape[0]):
            for j in range(self.coordinates.shape[1]):
                # Perturb coordinates positively
                self.coordinates[i, j] += delta
                positive_correlation = self.correlation()

                # Perturb coordinates negatively
                self.coordinates[i, j] -= 2 * delta
                negative_correlation = self.correlation()

                # Compute numerical derivative
                num_derivatives[i, j] = (positive_correlation - negative_correlation) / (2 * delta)

                # Reset coordinates to original value
                self.coordinates[i, j] += delta
            for j in range(self.sigma.shape[1]):
                # Perturb coordinates positively
                self.sigma[i, j] += delta
                positive_correlation = self.correlation()

                # Perturb coordinates negatively
                self.sigma[i, j] -= 2 * delta
                negative_correlation = self.correlation()

                # Compute numerical derivative
                num_derivatives[i, j + 3] = (positive_correlation - negative_correlation) / (2 * delta)

                # Reset coordinates to original value
                self.sigma[i, j] += delta
            self.epsilon[i] += delta
            positive_correlation = self.correlation()
            self.epsilon[i] -= 2 * delta
            negative_correlation = self.correlation()
            num_derivatives[i, 6] = (positive_correlation - negative_correlation) / (2 * delta)

        return num_derivatives

    def fit(self, n_iter=1000, learning_rate=0.1, tol=1e-5, numerical=False, verbose=True):
        """Maximize the correlation coefficient by gradient ascent on coordinates.

        At each iteration the coordinate gradient of the correlation is computed
        and a step of size ``learning_rate / max(|grad|)`` is taken; coordinates
        are wrapped into the periodic box. Progress is reported via ``logging``
        (enable INFO-level logging to see it).

        Parameters
        ----------
        n_iter : int, optional
            Maximum number of iterations. Default 1000.
        learning_rate : float, optional
            Step scale; the actual step is ``learning_rate / max(|grad|)``.
        tol : float, optional
            Stop early when the change in correlation between iterations drops
            below this value. Set to 0 to always run ``n_iter`` iterations.
        numerical : bool, optional
            Use finite-difference gradients instead of the analytical ones
            (much slower; for debugging).
        verbose : bool, optional
            Log the correlation every 10 iterations and on convergence.

        Returns
        -------
        dict
            ``{"correlation", "n_iter", "converged", "history"}`` where
            ``history`` is the per-iteration correlation as an array.
        """
        box = self.voxel_size * self.n_voxels
        cc = self.correlation()
        history = [cc]
        converged = False
        i = 0
        for i in range(n_iter):
            dx = (self.gradient_numerical() if numerical else self.gradient())[:, :3]
            max_grad = np.abs(dx).max()
            if max_grad == 0:
                logging.info("fit: zero gradient at iteration %d; stopping.", i)
                break
            self.coordinates = np.mod(self.coordinates + (learning_rate / max_grad) * dx, box)
            new_cc = self.correlation()
            history.append(new_cc)
            if verbose and i % 10 == 0:
                logging.info("fit: iteration %d, cc = %.4f", i, new_cc)
            if abs(new_cc - cc) < tol:
                converged = True
                cc = new_cc
                if verbose:
                    logging.info("fit: converged at iteration %d (|dcc| < %g), cc = %.4f", i, tol, cc)
                break
            cc = new_cc
        return {"correlation": cc, "n_iter": i + 1, "converged": converged, "history": np.asarray(history)}

    def dsim_map_numerical(self, delta=1e-5):
        num_particles = self.coordinates.shape[0]
        sim_map_shape = self.simulation_map().shape
        derivatives = {
            "dx": np.zeros((num_particles,) + sim_map_shape),
            "dy": np.zeros((num_particles,) + sim_map_shape),
            "dz": np.zeros((num_particles,) + sim_map_shape),
        }

        for i in range(num_particles):
            for j, direction in enumerate(["dx", "dy", "dz"]):
                original_coordinate = self.coordinates[i, j]

                # Perturb coordinate in the positive direction
                self.coordinates[i, j] = original_coordinate + delta
                positive_sim_map = self.simulation_map()

                # Perturb coordinate in the negative direction
                self.coordinates[i, j] = original_coordinate - delta
                negative_sim_map = self.simulation_map()

                # Compute numerical derivative for this particle and direction
                derivatives[direction][i] = (positive_sim_map - negative_sim_map) / (2 * delta)

                # Reset coordinate to original value
                self.coordinates[i, j] = original_coordinate

        return derivatives

    @staticmethod
    def outer_mult(x, y, z):
        return x[:, None, None, :] * y[:, None, :, None] * z[:, :, None, None]

    def dsim_map(self):
        sigma = self.sigma * np.sqrt(2)

        x_mu_sigma = (self.voxel_limits[0] - self.coordinates[:, None, 0]) / sigma[:, None, 0]
        y_mu_sigma = (self.voxel_limits[1] - self.coordinates[:, None, 1]) / sigma[:, None, 1]
        z_mu_sigma = (self.voxel_limits[2] - self.coordinates[:, None, 2]) / sigma[:, None, 2]

        phix = (1 + erf(x_mu_sigma)) / 2
        phiy = (1 + erf(y_mu_sigma)) / 2
        phiz = (1 + erf(z_mu_sigma)) / 2

        dphix_dx = -np.exp(-(x_mu_sigma**2)) / np.sqrt(np.pi) / sigma[:, None, 0]
        dphiy_dy = -np.exp(-(y_mu_sigma**2)) / np.sqrt(np.pi) / sigma[:, None, 1]
        dphiz_dz = -np.exp(-(z_mu_sigma**2)) / np.sqrt(np.pi) / sigma[:, None, 2]

        dphix_ds = x_mu_sigma * dphix_dx * np.sqrt(2)
        dphiy_ds = y_mu_sigma * dphiy_dy * np.sqrt(2)
        dphiz_ds = z_mu_sigma * dphiz_dz * np.sqrt(2)

        dphix = phix[:, 1:] - phix[:, :-1]
        dphiy = phiy[:, 1:] - phiy[:, :-1]
        dphiz = phiz[:, 1:] - phiz[:, :-1]

        ddphix_dx = dphix_dx[:, 1:] - dphix_dx[:, :-1]
        ddphiy_dy = dphiy_dy[:, 1:] - dphiy_dy[:, :-1]
        ddphiz_dz = dphiz_dz[:, 1:] - dphiz_dz[:, :-1]

        ddphix_ds = dphix_ds[:, 1:] - dphix_ds[:, :-1]
        ddphiy_ds = dphiy_ds[:, 1:] - dphiy_ds[:, :-1]
        ddphiz_ds = dphiz_ds[:, 1:] - dphiz_ds[:, :-1]

        dsim = {}

        dsim["dx"] = self.outer_mult(ddphix_dx, dphiy, dphiz)
        dsim["dy"] = self.outer_mult(dphix, ddphiy_dy, dphiz)
        dsim["dz"] = self.outer_mult(dphix, dphiy, ddphiz_dz)
        dsim["dsx"] = self.outer_mult(ddphix_ds, dphiy, dphiz)
        dsim["dsy"] = self.outer_mult(dphix, ddphiy_ds, dphiz)
        dsim["dsz"] = self.outer_mult(dphix, dphiy, ddphiz_ds)
        dsim["eps"] = self.outer_mult(dphix, dphiy, dphiz)

        for key in dsim:
            dsim[key] = self.fold_padding(dsim[key])
        return dsim

    def gradient_numpy(self):
        dsim = self.dsim_map()
        dsim = np.array(
            [dsim["dx"], dsim["dy"], dsim["dz"], dsim["dsx"], dsim["dsy"], dsim["dsz"], dsim["eps"]]
        ).transpose(1, 0, 2, 3, 4)
        sim_raw = self.simulation_map()
        sim_raw_mean = sim_raw.mean()
        sim_raw_std = sim_raw.std()
        sim = (sim_raw - sim_raw_mean) / sim_raw_std
        exp = self.experimental_map
        cc = np.mean(sim * exp)
        num1 = np.mean(dsim * exp[None, None, :, :, :], axis=(2, 3, 4))
        num2 = np.mean(dsim * sim[None, None, :, :, :], axis=(2, 3, 4))

        # Final equation
        return (num1 - cc * num2) / sim_raw_std

    def gradient(self, device="cpu"):
        """Analytical gradient of the correlation coefficient.

        Parameters
        ----------
        device : {"cpu"}, optional
            Compute backend. Only the CPU Numba kernel is available today; a
            ``"cuda"`` kernel arrives in the performance phase.

        Returns
        -------
        numpy.ndarray, shape (n, 7)
            Per-particle derivatives of the correlation with respect to
            ``(x, y, z, sigma_x, sigma_y, sigma_z, epsilon)``. Columns ``0:3``
            are the coordinate gradient used to drive the fit/force.
        """
        if device != "cpu":
            raise NotImplementedError(
                f"device={device!r} is not available yet; only 'cpu' is supported "
                "(GPU kernels arrive in the performance phase)."
            )
        return dcorr_v3(
            self.coordinates,
            self.n_voxels,
            self.voxel_size,
            self.sigma,
            self.epsilon,
            self.experimental_map,
            self.padding,
            5,
        )

    def test(self):
        assert np.allclose(self.simulation_map(), self.simulation_map_numpy(), atol=1e-5)
        assert np.allclose(self.dsim_map()["dx"], self.dsim_map_numerical()["dx"], atol=1e-6)
        assert np.allclose(self.dsim_map()["dy"], self.dsim_map_numerical()["dy"], atol=1e-6)
        assert np.allclose(self.dsim_map()["dz"], self.dsim_map_numerical()["dz"], atol=1e-6)
        assert np.allclose(self.gradient()[:, :3], self.gradient_numerical()[:, :3])
        assert np.allclose(self.gradient_numpy()[:, :3], self.gradient_numerical()[:, :3])
        assert np.allclose(self.gradient()[:, :3], self.gradient_numpy()[:, :3], atol=1e-7)
        assert np.allclose(self.gradient()[:, 3:], self.gradient_numpy()[:, 3:], atol=1e-5)
        assert np.allclose(
            dcorr_v3(
                self.coordinates,
                self.n_voxels,
                self.voxel_size,
                self.sigma,
                self.epsilon,
                self.experimental_map,
                self.padding,
                5,
            ),
            self.gradient_numpy(),
            atol=1e-5,
        )
