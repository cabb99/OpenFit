"""Tests for the :class:`openfit.Fit` density-fitting engine.

These formalize the assertions that previously lived in the ``Fit.test()``
method and the module ``__main__`` block: the Numba-accelerated kernels must
agree with the pure-NumPy implementations, and the analytical derivatives must
agree with finite-difference references.
"""

import numpy as np
import pytest

from openfit import Fit


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

def test_dcorr_coef_analytical_vs_numerical(small_fit):
    """Numba ``dcorr_v3`` coordinate gradient matches finite differences."""
    assert np.allclose(
        small_fit.dcorr_coef()[:, :3],
        small_fit.dcorr_coef_numerical()[:, :3],
    )


def test_dcorr_coef_numpy_vs_numerical(small_fit):
    """Pure-NumPy correlation gradient matches finite differences."""
    assert np.allclose(
        small_fit.dcorr_coef_numpy()[:, :3],
        small_fit.dcorr_coef_numerical()[:, :3],
    )


def test_dcorr_coef_v3_vs_numpy_coordinates(small_fit):
    """Numba and NumPy correlation gradients agree for coordinates."""
    assert np.allclose(
        small_fit.dcorr_coef()[:, :3],
        small_fit.dcorr_coef_numpy()[:, :3],
        atol=1e-7,
    )


def test_dcorr_coef_v3_vs_numpy_sigma(small_fit):
    """Numba and NumPy correlation gradients agree for sigma."""
    assert np.allclose(
        small_fit.dcorr_coef()[:, 3:],
        small_fit.dcorr_coef_numpy()[:, 3:],
        atol=1e-5,
    )


# --- Correlation coefficient ---------------------------------------------

def test_corr_coef_self_is_one():
    """Fitting a map against the density that produced it gives cc ~= 1."""
    nx, ny, nz = 30, 25, 20
    rng = np.random.default_rng(1)
    coords = rng.random((8, 3)) * (nx, ny, nz)
    sigma = np.ones((8, 3)) * 2.0
    epsilon = np.ones(8)

    # Generate a reference density, then fit an identical system to it.
    reference = Fit(np.zeros((nz, ny, nx)), voxel_size=[1, 1, 1])
    reference.set_coordinates(coords, sigma, epsilon)
    target_map = reference.simulation_map()

    fit = Fit(target_map, voxel_size=[1, 1, 1])
    fit.set_coordinates(coords, sigma, epsilon)
    assert fit.corr_coef() == pytest.approx(1.0, abs=1e-6)


def test_corr_coef_in_range(small_fit):
    """The correlation coefficient is bounded in [-1, 1]."""
    cc = small_fit.corr_coef()
    assert -1.0 <= cc <= 1.0


# --- Constructors --------------------------------------------------------

def test_from_dimensions_shape_and_voxel():
    """``from_dimensions`` produces a map covering the requested bounds."""
    fit = Fit.from_dimensions(
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
        Fit(np.zeros((10, 10)), voxel_size=[1, 1, 1])


def test_constructor_rejects_bad_voxel_size():
    with pytest.raises(ValueError):
        Fit(np.zeros((5, 5, 5)), voxel_size=[1, 1])


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

    reloaded = Fit.from_mrc(str(path))
    assert reloaded.experimental_map.shape == small_fit.experimental_map.shape
    # Both maps are normalized on load, so they should match closely
    # (float32 storage in the MRC file sets the tolerance).
    assert np.allclose(
        reloaded.experimental_map,
        small_fit.experimental_map,
        atol=1e-4,
    )
    assert np.allclose(reloaded.voxel_size, small_fit.voxel_size, atol=1e-4)
