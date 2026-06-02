"""Tests for the high-level :class:`openfit.Fit` orchestrator."""

from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from openfit import DensityMap, Fit  # noqa: E402  (after importorskip)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _tiny_system(n=5):
    """A small free-particle system + topology + positions + matching density."""
    from openmm import app, unit

    rng = np.random.default_rng(0)
    density = DensityMap(rng.random((20, 20, 20)), voxel_size=[1, 1, 1])
    density.set_coordinates(rng.uniform(2, 8, (n, 3)), np.full((n, 3), 2.0), np.ones(n))

    system = openmm.System()
    for _ in range(n):
        system.addParticle(12.0)
    topology = app.Topology()
    residue = topology.addResidue("UNK", topology.addChain())
    for i in range(n):
        topology.addAtom(f"P{i}", app.Element.getBySymbol("C"), residue)
    positions = (density.coordinates / 10.0) * unit.nanometer
    return topology, system, positions, density


def test_from_system_refines_and_reports_cc():
    topology, system, positions, density = _tiny_system()
    fit = Fit.from_system(topology, system, positions, density)

    assert fit.simulation.system.getNumForces() >= 1
    result = fit.refine(steps=40, record_interval=20)
    assert set(result) == {"correlation", "steps", "history"}
    assert result["history"].shape == (2,)
    assert np.isfinite(result["correlation"])
    assert np.isfinite(fit.cc)


def test_dock_runs_and_keeps_finite_cc():
    topology, system, positions, density = _tiny_system()
    fit = Fit.from_system(topology, system, positions, density)
    result = fit.dock(n_rotations=12, n_seeds=2, refine_iters=20, seed=0)
    assert set(result) == {"coordinates", "rotation", "translation", "cc"}
    assert np.isfinite(fit.cc)


def test_rigid_search_opt_in_on_builder():
    topology, system, positions, density = _tiny_system()
    fit = Fit.from_system(
        topology,
        system,
        positions,
        density,
        rigid_search={"n_rotations": 12, "n_seeds": 2, "refine_iters": 20, "seed": 0},
    )
    assert np.isfinite(fit.cc)


def test_native_backend_not_implemented():
    topology, system, positions, density = _tiny_system()
    with pytest.raises(NotImplementedError):
        Fit.from_system(topology, system, positions, density, backend="native")


def test_unknown_backend_raises():
    topology, system, positions, density = _tiny_system()
    with pytest.raises(ValueError):
        Fit.from_system(topology, system, positions, density, backend="bogus")


def test_save_and_save_map(tmp_path):
    pytest.importorskip("mrcfile")
    topology, system, positions, density = _tiny_system()
    fit = Fit.from_system(topology, system, positions, density)
    fit.refine(steps=20)

    pdb = tmp_path / "out.pdb"
    mrc = tmp_path / "out.mrc"
    fit.save(str(pdb))
    fit.save_map(str(mrc))
    assert pdb.exists() and pdb.stat().st_size > 0
    assert mrc.exists() and mrc.stat().st_size > 0


def test_from_smog_short_run():
    """End-to-end with the committed 4AKE model (skips without OpenSMOG)."""
    pytest.importorskip("OpenSMOG")
    data = EXAMPLES / "4ake"
    for name in ("4ake.AA.gro", "4ake.AA.top", "4ake.AA.xml", "1AKE.mrc"):
        if not (data / name).exists():
            pytest.skip(f"missing 4ake input {name}")

    fit = Fit.from_smog(
        data / "4ake.AA.gro",
        data / "4ake.AA.top",
        data / "4ake.AA.xml",
        data / "1AKE.mrc",
        update_interval=50,
    )
    cc0 = fit.cc
    result = fit.refine(steps=100, record_interval=50)
    assert np.isfinite(cc0)
    assert np.isfinite(result["correlation"])
    assert result["history"].shape == (2,)
