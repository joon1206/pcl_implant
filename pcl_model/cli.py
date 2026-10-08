"""Command-line interface for simulation and CAD inspection."""

from __future__ import annotations

import argparse
import json

from .geometry import inspect_mesh
from .io import export_result, load_configuration
from .solver import simulate
from .visualization import plot_profiles, plot_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCL implant degradation model")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="run a configured 1D degradation simulation")
    run.add_argument("--config", default="configs/pcl_hydrolysis.yaml")
    run.add_argument("--outdir", default="results/simulation")
    mesh = commands.add_parser("mesh", help="audit an STL/mesh without implying a field solve")
    mesh.add_argument("path")
    mesh.add_argument("--unit", choices=["mm", "cm", "m", "in"], default="mm")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "mesh":
        print(json.dumps(inspect_mesh(args.path, args.unit).as_dict(), indent=2))
        return
    parameters, config = load_configuration(args.config)
    result = simulate(parameters, config)
    export_result(result, parameters, config, args.outdir)
    plot_summary(result, args.outdir)
    plot_profiles(result, args.outdir)
    print(json.dumps({"outdir": args.outdir, "failure_time_days": result.failure_time_days, **result.diagnostics}, indent=2))


if __name__ == "__main__":
    main()

