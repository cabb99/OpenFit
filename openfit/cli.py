"""Command-line interface for OpenFit.

Two entry points (the ``openfit`` console script):

* ``openfit refine`` — flexible-fit a structure into a density map from flags.
* ``openfit run config.yaml`` — the same, driven by a YAML config file.

Both build a high-level :class:`openfit.Fit` from one structure source
(``--pdb`` Amber, ``--charmm``, ``--awsem``, ``--smog`` prebuilt model, or
``--smog-structure`` to generate one with SMOG 2), run
:meth:`~openfit.Fit.refine`, and write the result.
"""

import argparse
import sys


def _run(config):
    """Execute one refinement described by a config dict. Returns an exit code."""
    from openfit import Fit

    map_path = config.get("map")
    output = config.get("output")
    if not map_path:
        raise SystemExit("error: no target 'map' given")
    if not output:
        raise SystemExit("error: no 'output' PDB path given")

    sources = {key: config.get(key) for key in ("pdb", "charmm", "awsem", "smog", "smog_structure")}
    chosen = [(key, value) for key, value in sources.items() if value]
    if len(chosen) != 1:
        raise SystemExit("error: specify exactly one structure source: pdb / charmm / awsem / smog / smog_structure")
    source, value = chosen[0]

    common = dict(
        k=float(config.get("k", 6400)),
        update_interval=int(config.get("update_interval", 50)),
        platform=config.get("platform"),
        rigid_search=config.get("rigid_search", False),
    )
    if config.get("sigma") is not None:
        common["sigma"] = float(config["sigma"])

    if source == "smog":
        gro, top, xml = value
        fit = Fit.from_smog(gro, top, xml, map_path, platform=common.pop("platform") or "CPU", **common)
    elif source == "smog_structure":
        fit = Fit.from_smog_structure(
            value, map_path, model=config.get("smog_model", "AA"), platform=common.pop("platform") or "CPU", **common
        )
    elif source == "awsem":
        fit = Fit.from_awsem(value, map_path, **common)
    elif source == "charmm":
        fit = Fit.from_charmm(value, map_path, backend=config.get("backend", "python"), **common)
    else:  # pdb -> Amber
        fit = Fit.from_amber(value, map_path, backend=config.get("backend", "python"), **common)

    print(f"initial correlation: {fit.cc:.4f}", flush=True)
    result = fit.refine(
        steps=int(config.get("steps", 50000)),
        minimize=bool(config.get("minimize", False)),
        trajectory=config.get("trajectory"),
        trajectory_interval=int(config.get("trajectory_interval", 1000)),
    )
    fit.save(output)
    if config.get("output_map"):
        fit.save_map(config["output_map"])
    print(f"final correlation:   {result['correlation']:.4f}", flush=True)
    print(f"wrote {output}" + (f" and {config['output_map']}" if config.get("output_map") else ""), flush=True)
    return 0


def cmd_refine(args):
    config = {
        "map": args.map,
        "output": args.output,
        "output_map": args.output_map,
        "trajectory": args.trajectory,
        "trajectory_interval": args.trajectory_interval,
        "pdb": args.pdb,
        "charmm": args.charmm,
        "awsem": args.awsem,
        "smog": args.smog,
        "smog_structure": args.smog_structure,
        "smog_model": args.smog_model,
        "steps": args.steps,
        "k": args.k,
        "update_interval": args.update_interval,
        "sigma": args.sigma,
        "platform": args.platform,
        "backend": args.backend,
        "minimize": args.minimize,
        "rigid_search": args.rigid_search,
    }
    return _run(config)


def cmd_run(args):
    try:
        import yaml
    except ImportError:
        raise SystemExit("error: 'openfit run' needs PyYAML — install with: pip install \"openfit[cli]\"")
    with open(args.config) as handle:
        config = yaml.safe_load(handle) or {}
    return _run(config)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openfit",
        description="Flexibly fit a molecular structure into a 3D density map.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    refine = sub.add_parser("refine", help="fit a structure into a density map")
    source = refine.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb", metavar="FILE", help="all-atom PDB (Amber14 + implicit solvent)")
    source.add_argument("--charmm", metavar="FILE", help="all-atom PDB/CIF (CHARMM36)")
    source.add_argument("--awsem", metavar="FILE", help="PDB/CIF, coarse-grained OpenAWSEM model")
    source.add_argument("--smog", nargs=3, metavar=("GRO", "TOP", "XML"), help="prebuilt OpenSMOG model files")
    source.add_argument("--smog-structure", metavar="FILE", help="PDB/CIF; generate a SMOG model with SMOG 2")
    refine.add_argument("map", help="target density map (MRC/CCP4)")
    refine.add_argument("-o", "--output", required=True, metavar="PDB", help="output structure (PDB)")
    refine.add_argument("--output-map", metavar="MRC", help="also write the fitted density to this MRC file")
    refine.add_argument("--trajectory", metavar="DCD", help="write the MD trajectory to this DCD file")
    refine.add_argument("--trajectory-interval", type=int, default=1000, help="steps between trajectory frames")
    refine.add_argument(
        "--smog-model", default="AA", choices=["AA", "CA", "AAgaussian", "CAgaussian"], help="SMOG model type"
    )
    refine.add_argument("--steps", type=int, default=50000, help="MD steps to run (default 50000)")
    refine.add_argument("--k", type=float, default=6400, help="density force constant (default 6400)")
    refine.add_argument("--update-interval", type=int, default=50, help="steps between force refreshes")
    refine.add_argument("--sigma", type=float, default=None, help="Gaussian width in Angstrom (builder default)")
    refine.add_argument("--platform", default=None, help="OpenMM platform (CUDA/OpenCL/CPU/Reference)")
    refine.add_argument("--backend", default="python", choices=["python", "native"], help="force backend")
    refine.add_argument("--minimize", action="store_true", help="energy-minimize before the run")
    refine.add_argument(
        "--rigid-search", action="store_true", help="rigid-body dock (orientation+translation scan) before refining"
    )
    refine.set_defaults(func=cmd_refine)

    run = sub.add_parser("run", help="run a refinement from a YAML config file")
    run.add_argument("config", help="YAML configuration file")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
