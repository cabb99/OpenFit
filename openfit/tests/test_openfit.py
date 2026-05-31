"""
Unit and regression test for the openfit package.
"""

# Import package, test suite, and other packages as needed
import sys

import openfit


def test_openfit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openfit" in sys.modules


def test_public_api():
    """The documented public names are importable from the top-level package."""
    assert hasattr(openfit, "Fit")
    assert hasattr(openfit, "generate_rotations")


def test_version_is_a_string():
    assert isinstance(openfit.__version__, str)
    assert openfit.__version__
