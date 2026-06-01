"""Smoke tests for the scripts in ``examples/``.

Each example exposes a ``main()`` function. Example 01 has no optional
dependencies and is run end-to-end; the others are guarded with
``importorskip`` so they skip when OpenMM / mrcfile / MolScene are unavailable.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load(example_name):
    path = EXAMPLES_DIR / example_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_01_fit_synthetic():
    module = _load("01_fit_synthetic.py")
    initial_cc, final_cc = module.main(n_steps=50)
    assert final_cc > initial_cc
    assert final_cc > 0.9


def test_example_02_fit_from_mrc():
    pytest.importorskip("mrcfile")
    module = _load("02_fit_from_mrc.py")
    initial_cc, final_cc = module.main()
    assert final_cc > initial_cc


def test_example_03_pdb_to_density(tmp_path):
    pytest.importorskip("molscene")
    module = _load("03_pdb_to_density.py")
    out = module.main()
    assert Path(out).exists()


def test_example_04_openmm_integration():
    pytest.importorskip("openmm")
    module = _load("04_openmm_integration.py")
    cc_start, cc_end = module.main(total_steps=100, update_every=50)
    assert np.isfinite(cc_start) and np.isfinite(cc_end)
