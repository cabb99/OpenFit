"""Tests for :func:`openfit.generate_rotations`.

``generate_rotations`` returns a :class:`scipy.spatial.transform.Rotation`
built from unit quaternions, either from the vertices of a regular 4-polytope
(for special values of ``n``) or from random/optimized quaternions otherwise.
Regardless of the path, the result must be a set of valid proper rotations.
"""

import numpy as np
import pytest

from openfit import generate_rotations


# Values that map to regular-polytope vertex generators, plus a couple of
# arbitrary counts that exercise the random path. Optimization is left off
# (optimize=0) to keep the tests fast and deterministic-ish.
@pytest.mark.parametrize("n", [4, 5, 8, 12, 60, 7, 50])
def test_generate_rotations_are_valid(n):
    rotations = generate_rotations(n, optimize=0)
    matrices = rotations.as_matrix()

    assert matrices.ndim == 3
    assert matrices.shape[1:] == (3, 3)
    assert len(matrices) >= 1

    identity = np.eye(3)
    for m in matrices:
        # Orthonormal columns.
        assert np.allclose(m @ m.T, identity, atol=1e-6)
        # Proper rotation (no reflection).
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n", [7, 50])
def test_random_path_returns_requested_count(n):
    """For non-polytope ``n``, exactly ``n`` rotations are returned."""
    rotations = generate_rotations(n, optimize=0)
    assert len(rotations.as_quat()) == n


def test_quaternions_are_unit_norm():
    rotations = generate_rotations(50, optimize=0)
    norms = np.linalg.norm(rotations.as_quat(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
