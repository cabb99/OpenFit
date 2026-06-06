"""Tests for the :class:`openfit.DensityMap` density-fitting engine.

These formalize the assertions that previously lived in the ``Fit.test()``
method and the module ``__main__`` block: the Numba-accelerated kernels must
agree with the pure-NumPy implementations, and the analytical derivatives must
agree with finite-difference references.
"""

import numpy as np
import pytest

from openfit import DensityMap

# --- Simulated map -------------------------------------------------------


def test_sim_map_matches_numpy(small_fit):
    """The Numba ``sim_map`` kernel matches the pure-NumPy reference."""
    assert np.allclose(
        small_fit.simulation_map(),
        small_fit.simulation_map_numpy(),
        atol=1e-5,
    )


def test_simulation_map_shape(small_fit):
    """The simulated map has the experimental map's shape."""
    assert small_fit.simulation_map().shape == small_fit.experimental_map.shape


# --- Derivatives of the simulated map ------------------------------------


@pytest.mark.parametrize("axis", ["dx", "dy", "dz"])
def test_dsim_map_matches_numerical(small_fit, axis):
    """Analytical d(sim_map)/d(coord) matches finite differences."""
    analytical = small_fit.dsim_map()[axis]
    numerical = small_fit.dsim_map_numerical()[axis]
    assert np.allclose(analytical, numerical, atol=1e-6)


# --- Correlation-coefficient gradient ------------------------------------


def test_gradient_analytical_vs_numerical(small_fit):
    """Numba ``dcorr_v3`` coordinate gradient matches finite differences."""
    assert np.allclose(
        small_fit.gradient()[:, :3],
        small_fit.gradient_numerical()[:, :3],
    )


def test_gradient_numpy_vs_numerical(small_fit):
    """Pure-NumPy correlation gradient matches finite differences."""
    assert np.allclose(
        small_fit.gradient_numpy()[:, :3],
        small_fit.gradient_numerical()[:, :3],
    )


def test_gradient_v3_vs_numpy_coordinates(small_fit):
    """Numba and NumPy correlation gradients agree for coordinates."""
    assert np.allclose(
        small_fit.gradient()[:, :3],
        small_fit.gradient_numpy()[:, :3],
        atol=1e-7,
    )


def test_gradient_v3_vs_numpy_sigma(small_fit):
    """Numba and NumPy correlation gradients agree for sigma."""
    assert np.allclose(
        small_fit.gradient()[:, 3:],
        small_fit.gradient_numpy()[:, 3:],
        atol=1e-5,
    )


def test_force_gradient_matches_full_gradient(small_fit):
    """The fast force-only kernel matches dcorr_v3's coordinate gradient."""
    grad3, _cc = small_fit.force_gradient()
    assert grad3.shape == (small_fit.coordinates.shape[0], 3)
    assert np.allclose(grad3, small_fit.gradient()[:, :3], atol=1e-7)


def test_force_gradient_returns_correlation(small_fit):
    """force_gradient's cached cc matches the standalone correlation()."""
    _grad3, cc = small_fit.force_gradient()
    assert cc == pytest.approx(small_fit.correlation(), abs=1e-6)


def test_force_gradient_cuda_matches_cpu():
    """The on-device CUDA gradient matches the CPU kernel (skipped without a GPU)."""
    from openfit._cuda_kernels import CUDA_AVAILABLE

    if not CUDA_AVAILABLE:
        pytest.skip("no CUDA GPU / numba.cuda available")
    rng = np.random.default_rng(0)
    dm = DensityMap(rng.random((40, 40, 40)), voxel_size=[2, 2, 2])
    box = 80.0  # keep atoms away from the box edges (CUDA path truncates, CPU folds PBC)
    dm.set_coordinates(rng.uniform(0.25 * box, 0.75 * box, (50, 3)), np.full((50, 3), 2.0), np.ones(50))
    g_cpu, cc_cpu = dm.force_gradient(device="cpu")
    g_gpu, cc_gpu = dm.force_gradient(device="cuda")
    assert np.allclose(g_gpu, g_cpu, atol=1e-6)
    assert cc_gpu == pytest.approx(cc_cpu, abs=1e-6)


# --- Correlation coefficient ---------------------------------------------


def test_correlation_self_is_one():
    """Fitting a map against the density that produced it gives cc ~= 1."""
    nx, ny, nz = 30, 25, 20
    rng = np.random.default_rng(1)
    coords = rng.random((8, 3)) * (nx, ny, nz)
    sigma = np.ones((8, 3)) * 2.0
    epsilon = np.ones(8)

    # Generate a reference density, then fit an identical system to it.
    reference = DensityMap(np.zeros((nz, ny, nx)), voxel_size=[1, 1, 1])
    reference.set_coordinates(coords, sigma, epsilon)
    target_map = reference.simulation_map()

    fit = DensityMap(target_map, voxel_size=[1, 1, 1])
    fit.set_coordinates(coords, sigma, epsilon)
    assert fit.correlation() == pytest.approx(1.0, abs=1e-6)


def test_correlation_in_range(small_fit):
    """The correlation coefficient is bounded in [-1, 1]."""
    cc = small_fit.correlation()
    assert -1.0 <= cc <= 1.0


# --- rigid-body search ---------------------------------------------------


def test_rigid_fit_improves_from_scrambled_pose():
    """rigid_fit recovers a better placement of a rotated/displaced structure."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    n = 8
    coords = rng.uniform(12, 28, size=(n, 3))
    sigma = np.full((n, 3), 2.0)
    eps = np.ones(n)

    target = DensityMap(np.zeros((40, 40, 40)), voxel_size=[1, 1, 1])
    target.set_coordinates(coords, sigma, eps)
    gt_internal = target.coordinates.copy()  # ground-truth pose in the internal frame
    dm = DensityMap(target.simulation_map(), voxel_size=[1, 1, 1])

    rot = Rotation.random(random_state=1).as_matrix()
    centroid = coords.mean(0)
    scrambled = (coords - centroid) @ rot.T + centroid + np.array([5.0, -4.0, 3.0])
    dm.set_coordinates(scrambled, sigma, eps)
    before = dm.correlation()

    # n_rotations=300 (600-cell) is deterministic; seed fixes the refinement.
    best = dm.rigid_fit(n_rotations=300, n_seeds=5, refine_iters=150, seed=0)
    assert set(best) == {"coordinates", "rotation", "translation", "cc"}
    assert best["coordinates"].shape == (n, 3)
    assert best["rotation"].shape == (3, 3)
    assert best["cc"] > before
    assert best["cc"] > 0.95  # coarse scan + local refinement recovers the pose

    # the engine is left at the best pose, and it matches the ground truth closely
    assert dm.correlation() == pytest.approx(best["cc"], abs=1e-6)
    rmsd = np.sqrt(((best["coordinates"] - gt_internal) ** 2).sum(1).mean())
    assert rmsd < 2.0


def test_rigid_fit_requires_coordinates():
    dm = DensityMap(np.zeros((10, 10, 10)), voxel_size=[1, 1, 1])
    with pytest.raises(RuntimeError):
        dm.rigid_fit()


# --- fit() optimization loop ---------------------------------------------


def test_fit_improves_and_returns_history():
    """fit() raises the correlation and returns a convergence summary."""
    nx, ny, nz = 30, 25, 20
    rng = np.random.default_rng(2)
    coords = rng.uniform(5, 15, size=(6, 3))
    sigma = np.full((6, 3), 2.0)
    epsilon = np.ones(6)

    reference = DensityMap(np.zeros((nz, ny, nx)), voxel_size=[1, 1, 1])
    reference.set_coordinates(coords, sigma, epsilon)
    target = reference.simulation_map()

    fit = DensityMap(target, voxel_size=[1, 1, 1])
    fit.set_coordinates(coords + rng.normal(scale=1.0, size=(6, 3)), sigma, epsilon)
    initial = fit.correlation()

    result = fit.fit(n_iter=100, tol=0.0, verbose=False)
    assert set(result) == {"correlation", "n_iter", "converged", "history"}
    assert result["correlation"] > initial
    assert result["history"].shape == (result["n_iter"] + 1,)


# --- repr ----------------------------------------------------------------


def test_repr(small_fit):
    text = repr(small_fit)
    assert text.startswith("DensityMap(")
    assert "particles=10" in text


# --- from_scene (MolScene bridge) ----------------------------------------


def test_from_scene():
    """Build a Fit from a MolScene Scene (skips if MolScene is unavailable)."""
    molscene = pytest.importorskip("molscene")
    pdb = (
        "ATOM      1  N   GLY A   1      11.104   6.134   7.123  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1      12.560   6.087   7.220  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1      13.000   4.700   7.660  1.00  0.00           C\n"
        "END\n"
    )
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(pdb)
        path = fh.name
    scene = molscene.Scene.from_pdb(path)
    fit = DensityMap.from_scene(scene, voxel_size=[1, 1, 1], padding=5.0)
    assert fit.coordinates.shape == (3, 3)
    assert np.all(np.asarray(fit.simulation_map().shape) > 0)


# --- Constructors --------------------------------------------------------


def test_from_dimensions_shape_and_voxel():
    """``from_dimensions`` produces a map covering the requested bounds."""
    fit = DensityMap.from_dimensions(
        min_coords=[0, 0, 0],
        max_coords=[10, 20, 30],
        voxel_size=[1, 1, 1],
    )
    # n_voxels is stored x, y, z; the map array is (z, y, x).
    assert tuple(fit.n_voxels) == (10, 20, 30)
    assert fit.experimental_map.shape == (30, 20, 10)
    assert np.allclose(fit.voxel_size, [1, 1, 1])


# --- Input validation ----------------------------------------------------


def test_constructor_rejects_non_3d_map():
    with pytest.raises(ValueError):
        DensityMap(np.zeros((10, 10)), voxel_size=[1, 1, 1])


def test_constructor_rejects_bad_voxel_size():
    with pytest.raises(ValueError):
        DensityMap(np.zeros((5, 5, 5)), voxel_size=[1, 1])


def test_set_coordinates_rejects_bad_coordinate_shape(small_fit):
    with pytest.raises(ValueError):
        small_fit.set_coordinates(np.zeros((5, 2)))


def test_set_coordinates_rejects_bad_sigma_shape(small_fit):
    with pytest.raises(ValueError):
        small_fit.set_coordinates(np.zeros((5, 3)), sigma=np.ones((5, 2)))


def test_set_coordinates_rejects_bad_epsilon_shape(small_fit):
    with pytest.raises(ValueError):
        small_fit.set_coordinates(np.zeros((5, 3)), epsilon=np.ones(4))


# --- MRC round-trip ------------------------------------------------------


def test_mrc_roundtrip(small_fit, tmp_path):
    """Saving and reloading the experimental map preserves it."""
    pytest.importorskip("mrcfile")
    path = tmp_path / "map.mrc"
    small_fit.save_mrc(str(path), experimental=True)

    reloaded = DensityMap.from_mrc(str(path))
    assert reloaded.experimental_map.shape == small_fit.experimental_map.shape
    # Both maps are normalized on load, so they should match closely
    # (float32 storage in the MRC file sets the tolerance).
    assert np.allclose(
        reloaded.experimental_map,
        small_fit.experimental_map,
        atol=1e-4,
    )
    assert np.allclose(reloaded.voxel_size, small_fit.voxel_size, atol=1e-4)
