"""Command-line interface for OpenFit.

Two entry points (the ``openfit`` console script):

* ``openfit refine`` — flexible-fit a structure into a density map from flags.
* ``openfit run config.yaml`` — the same, driven by a YAML config file.

Both build a high-level :class:`openfit.Fit` (``from_smog`` for SMOG models,
``from_amber`` for a PDB), run :meth:`~openfit.Fit.refine`, and write the result.
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

    smog = config.get("smog")
    pdb = config.get("pdb")
    if bool(smog) == bool(pdb):
        raise SystemExit("error: specify exactly one structure source ('smog' or 'pdb')")

    k = float(config.get("k", 3200))
    update_interval = int(config.get("update_interval", 50))
    sigma = float(config.get("sigma", 1.5))
    platform = config.get("platform")

    if smog:
        gro, top, xml = smog
        fit = Fit.from_smog(
            gro,
            top,
            xml,
            map_path,
            sigma=sigma,
            k=k,
            update_interval=update_interval,
            platform=platform or "CPU",
        )
    else:
        fit = Fit.from_amber(
            pdb,
            map_path,
            sigma=sigma,
            k=k,
            update_interval=update_interval,
            platform=platform,
            backend=config.get("backend", "python"),
        )

    print(f"initial correlation: {fit.cc:.4f}", flush=True)
    result = fit.refine(steps=int(config.get("steps", 50000)), minimize=bool(config.get("minimize", False)))
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
        "smog": args.smog,
        "pdb": args.pdb,
        "steps": args.steps,
        "k": args.k,
        "update_interval": args.update_interval,
        "sigma": args.sigma,
        "platform": args.platform,
        "backend": args.backend,
        "minimize": args.minimize,
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
    source.add_argument("--smog", nargs=3, metavar=("GRO", "TOP", "XML"), help="OpenSMOG structure-based model files")
    refine.add_argument("map", help="target density map (MRC/CCP4)")
    refine.add_argument("-o", "--output", required=True, metavar="PDB", help="output structure (PDB)")
    refine.add_argument("--output-map", metavar="MRC", help="also write the fitted density to this MRC file")
    refine.add_argument("--steps", type=int, default=50000, help="MD steps to run (default 50000)")
    refine.add_argument("--k", type=float, default=3200, help="density force constant (default 3200)")
    refine.add_argument("--update-interval", type=int, default=50, help="steps between force refreshes")
    refine.add_argument("--sigma", type=float, default=1.5, help="Gaussian width in Angstrom (default 1.5)")
    refine.add_argument("--platform", default=None, help="OpenMM platform (CUDA/OpenCL/CPU/Reference)")
    refine.add_argument("--backend", default="python", choices=["python", "native"], help="force backend")
    refine.add_argument("--minimize", action="store_true", help="energy-minimize before the run")
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
