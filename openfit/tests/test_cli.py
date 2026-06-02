"""Tests for the ``openfit`` command-line interface."""

from pathlib import Path

import pytest

from openfit.cli import build_parser, main

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SMOG = [str(EXAMPLES / "4ake" / f) for f in ("4ake.AA.gro", "4ake.AA.top", "4ake.AA.xml")]
MAP = str(EXAMPLES / "4ake" / "1AKE.mrc")


# --- argument parsing (no heavy deps) ------------------------------------


def test_parser_refine_smog():
    args = build_parser().parse_args(["refine", "--smog", "a.gro", "b.top", "c.xml", "m.mrc", "-o", "out.pdb"])
    assert args.command == "refine"
    assert args.smog == ["a.gro", "b.top", "c.xml"]
    assert args.pdb is None
    assert args.map == "m.mrc"
    assert args.output == "out.pdb"


def test_parser_refine_requires_a_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["refine", "m.mrc", "-o", "out.pdb"])


def test_parser_sources_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["refine", "--pdb", "p.pdb", "--smog", "a", "b", "c", "m.mrc", "-o", "out.pdb"])


def test_parser_run_subcommand():
    args = build_parser().parse_args(["run", "config.yaml"])
    assert args.command == "run"
    assert args.config == "config.yaml"


# --- end-to-end (need OpenSMOG + the 4ake data) --------------------------


def _have_inputs():
    return all(Path(p).exists() for p in SMOG + [MAP])


def test_refine_smog_end_to_end(tmp_path):
    pytest.importorskip("OpenSMOG")
    if not _have_inputs():
        pytest.skip("missing 4ake inputs")
    out = tmp_path / "refined.pdb"
    rc = main(["refine", "--smog", *SMOG, MAP, "-o", str(out), "--steps", "60", "--update-interval", "30"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


def test_run_yaml_end_to_end(tmp_path):
    pytest.importorskip("OpenSMOG")
    pytest.importorskip("yaml")
    if not _have_inputs():
        pytest.skip("missing 4ake inputs")
    import yaml

    out = tmp_path / "out.pdb"
    config = {
        "smog": SMOG,
        "map": MAP,
        "output": str(out),
        "steps": 60,
        "update_interval": 30,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    rc = main(["run", str(cfg_path)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
