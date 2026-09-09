"""
Command-line interface.

    gputrace list-loaders
    gputrace list-scenarios
    gputrace analyze --loader alibaba2020 --input pai_task_table_sample.csv --out fit.json
    gputrace generate --scenario bursty_arrivals --fit fit.json --n-jobs 50000 --seed 1 --out trace.csv
    gputrace generate-all --fit fit.json --n-jobs 20000 --seed 1 --out-dir traces/
    gputrace export --input trace.csv --format k8s_sim --out-dir k8s_sim_input/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import schema
from .analysis import AnalysisResult, analyze
from .generators import WorkloadScenarioGenerator, get_scenario, list_scenarios
from .loaders import get_loader, list_loaders


def _cmd_list_loaders(args: argparse.Namespace) -> int:
    for name in list_loaders():
        print(name)
    return 0


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    for spec in list_scenarios():
        print(f"{spec.name:28s} {spec.stresses}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    loader = get_loader(args.loader)
    df = loader.load(args.input)
    result = analyze(df, segment_by=args.segment_by)
    result.to_json(args.out)
    print(f"fit {len(df)} jobs from {args.input!r} -> {args.out}", file=sys.stderr)
    for name, ff in result.fields.items():
        flag = " [INFINITE MEAN — sampling capped]" if ff.infinite_mean_warning else ""
        print(f"  {name:15s} best={ff.best}{flag}", file=sys.stderr)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    fit = AnalysisResult.from_json(args.fit) if args.fit else _reference_fit()
    spec = get_scenario(args.scenario)
    gen = WorkloadScenarioGenerator(fit, seed=args.seed)
    df = spec.fn(gen, args.n_jobs)
    schema.validate(df, strict=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} jobs ({args.scenario}) -> {args.out}", file=sys.stderr)
    return 0


def _cmd_generate_all(args: argparse.Namespace) -> int:
    fit = AnalysisResult.from_json(args.fit) if args.fit else _reference_fit()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in list_scenarios():
        gen = WorkloadScenarioGenerator(fit, seed=args.seed)
        df = spec.fn(gen, args.n_jobs)
        schema.validate(df, strict=True)
        path = out_dir / f"{spec.name}.csv"
        df.to_csv(path, index=False)
        print(f"  {spec.name:28s} -> {path} ({len(df)} jobs)", file=sys.stderr)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    if args.format != "k8s_sim":
        print(f"unknown export format {args.format!r}", file=sys.stderr)
        return 1
    from .exporters.k8s_sim import export_scenario

    df = pd.read_csv(args.input)
    export_scenario(df, Path(args.out_dir), n_nodes=args.n_nodes, gpus_per_node=args.gpus_per_node)
    print(f"exported k8s_sim input to {args.out_dir}", file=sys.stderr)
    return 0


def _reference_fit() -> AnalysisResult:
    from .generators._reference import build_reference_fit

    print("no --fit given, using built-in reference fit (see docs — not a substitute for a real trace)",
          file=sys.stderr)
    return build_reference_fit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gputrace", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list-loaders", help="list registered trace loaders").set_defaults(func=_cmd_list_loaders)
    sub.add_parser("list-scenarios", help="list registered stress-test scenarios").set_defaults(func=_cmd_list_scenarios)

    pa = sub.add_parser("analyze", help="fit distributions to a real source trace")
    pa.add_argument("--loader", required=True, choices=list_loaders() + ["synthetic"])
    pa.add_argument("--input", required=True)
    pa.add_argument("--out", required=True, help="path to write the fit as JSON")
    pa.add_argument("--segment-by", default="gpu_type")
    pa.set_defaults(func=_cmd_analyze)

    pg = sub.add_parser("generate", help="generate one synthetic scenario")
    pg.add_argument("--scenario", required=True)
    pg.add_argument("--fit", default=None, help="fit JSON from `analyze`; omit to use the built-in reference fit")
    pg.add_argument("--n-jobs", type=int, default=10000)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=_cmd_generate)

    pga = sub.add_parser("generate-all", help="generate every registered scenario")
    pga.add_argument("--fit", default=None)
    pga.add_argument("--n-jobs", type=int, default=10000)
    pga.add_argument("--seed", type=int, default=0)
    pga.add_argument("--out-dir", required=True)
    pga.set_defaults(func=_cmd_generate_all)

    pe = sub.add_parser("export", help="convert a unified-schema trace into a simulator's native input format")
    pe.add_argument("--input", required=True)
    pe.add_argument("--format", default="k8s_sim", choices=["k8s_sim"])
    pe.add_argument("--out-dir", required=True)
    pe.add_argument("--n-nodes", type=int, default=20)
    pe.add_argument("--gpus-per-node", type=int, default=8)
    pe.set_defaults(func=_cmd_export)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
