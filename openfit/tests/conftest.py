"""Shared pytest fixtures for the openfit test suite."""

import numpy as np
import pytest

from openfit import Fit


# A small, deterministic system. These dimensions mirror the historical
# ``Fit.test()`` / ``__main__`` sanity check, which is known to pass the
# analytical-vs-numerical derivative comparisons at the tolerances used below.
N_PARTICLES = 10
NX, NY, NZ = 70, 60, 50
SEED = 0


@pytest.fixture
def rng():
    """A seeded NumPy random generator for reproducible tests."""
    return np.random.default_rng(SEED)


@pytest.fixture
def coordinates(rng):
    """Random particle coordinates inside the map bounds, shape (n, 3)."""
    return rng.random((N_PARTICLES, 3)) * (NX, NY, NZ)


@pytest.fixture
def small_fit(rng, coordinates):
    """A configured :class:`~openfit.Fit` on a random experimental map.

    The map is random noise, which is sufficient for validating the
    analytical machinery (simulated map, derivatives, correlation gradient)
    against numerical references.
    """
    experimental_map = rng.random((NZ, NY, NX))
    sigma = np.ones((N_PARTICLES, 3))
    epsilon = np.ones(N_PARTICLES)
    fit = Fit(experimental_map, voxel_size=[1, 1, 1])
    fit.set_coordinates(coordinates, sigma, epsilon)
    return fit
