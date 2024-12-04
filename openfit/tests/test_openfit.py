"""
Unit and regression test for the openfit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openfit


def test_openfit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openfit" in sys.modules
