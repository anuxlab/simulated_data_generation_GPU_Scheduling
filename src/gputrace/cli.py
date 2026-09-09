"""
Command-line interface for gputrace.

    gputrace analyze data.csv --loader alibaba2020 --out fit_results.json
    gputrace generate fit_results.json --scenario bursty_arrivals --n-jobs 20000 --out trace.csv
    gputrace list-scenarios
    gputrace list-loaders
"""
from __future__ import annotations

import argparse
import json
import sys

from . import loaders
from .analysis import analyze as run_analysis
from .generators import WorkloadScenarioGenerator, list_scenarios, run_scenario


def cmd_analyze(args: argparse.Namespace) -> int:
    loader = loaders.get_loader(args.loader)
    df = loader.load(args.input, strict=args.strict)
    if len(df) == 0:
        print("no rows after loading/validation", file=sys.stderr)
        return 1
    results = run_analysis(df)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"analyzed {len(df):,} jobs from {args.input} ({args.loader}) -> {args.out}")
    for metric in ("duration", "num_cpu", "num_gpu", "wait_time", "inter_arrival"):
        if metric in results:
            r = results[metric]
            warn = f"  [!] {r['infinite_mean_warning']}" if r.get("infinite_mean_warning") else ""
            print(f"  {metric}: best_dist={r['best_dist']} n={r['stats']['n']}{warn}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    with open(args.fit_results) as f:
        fits = json.load(f)
    gen = WorkloadScenarioGenerator(fits, seed=args.seed)
    df = run_scenario(args.scenario, gen, args.n_jobs)
    df.to_csv(args.out, index=False)
    span_h = (df["submit_time"].max() - df["submit_time"].min()) / 3600
    print(f"scenario={args.scenario} n={len(df):,} span={span_h:.1f}h "
          f"mean_gpu={df['num_gpu'].mean():.2f} mean_cpu={df['num_cpu'].mean():.2f} "
          f"median_duration={df['duration'].median():.0f}s -> {args.out}")
    return 0


def cmd_generate_all(args: argparse.Namespace) -> int:
    with open(args.fit_results) as f:
        fits = json.load(f)
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name in list_scenarios():
        gen = WorkloadScenarioGenerator(fits, seed=args.seed)
        df = run_scenario(name, gen, args.n_jobs)
        out = f"{args.out_dir}/{name}.csv"
        df.to_csv(out, index=False)
        print(f"  {name}: {len(df):,} jobs -> {out}")
    return 0


def cmd_list_scenarios(_args: argparse.Namespace) -> int:
    for name, desc in list_scenarios().items():
        print(f"{name}\n    {desc}\n")
    return 0


def cmd_list_loaders(_args: argparse.Namespace) -> int:
    for name in loaders.available_loaders():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gputrace")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="fit distributions to a trace and write fit_results.json")
    a.add_argument("input")
    a.add_argument("--loader", default="alibaba2020", choices=loaders.available_loaders())
    a.add_argument("--strict", action="store_true")
    a.add_argument("--out", default="fit_results.json")
    a.set_defaults(func=cmd_analyze)

    g = sub.add_parser("generate", help="generate one synthetic stress-test trace")
    g.add_argument("fit_results")
    g.add_argument("--scenario", default="baseline")
    g.add_argument("--n-jobs", type=int, default=20000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="trace.csv")
    g.set_defaults(func=cmd_generate)

    ga = sub.add_parser("generate-all", help="generate every registered scenario at once")
    ga.add_argument("fit_results")
    ga.add_argument("--n-jobs", type=int, default=20000)
    ga.add_argument("--seed", type=int, default=42)
    ga.add_argument("--out-dir", default="synthetic_traces")
    ga.set_defaults(func=cmd_generate_all)

    ls = sub.add_parser("list-scenarios", help="list available stress-test scenarios")
    ls.set_defaults(func=cmd_list_scenarios)

    ll = sub.add_parser("list-loaders", help="list available dataset loaders")
    ll.set_defaults(func=cmd_list_loaders)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
