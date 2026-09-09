# gputrace

A source-agnostic framework for statistically analyzing datacenter/GPU cluster traces
(Alibaba `cluster-trace-gpu-v2020`, Google cluster trace, your own private trace, ...)
and generating synthetic **benchmark & stress-test workloads** for GPU-scheduling /
cloud resource-management algorithm evaluation.

```
 real trace (any source) --[loader]--> unified schema --[analyze]--> fit_results.json
                                                                          |
                                                                    [scenario generator]
                                                                          |
                                            synthetic stress-test traces (baseline,
                                            bursty arrivals, flash crowd, heavy-tail
                                            demand, diurnal, high contention, ...)
```

## Why

Trace-driven simulation is only as good as the traces you test against. Real traces
give you *one* realistic scenario; they don't tell you how your scheduler behaves
under a flash crowd, a heavy-tailed resource-hog workload, or a bursty non-Poisson
arrival process. `gputrace` fits real trace data to a distribution, is honest about
where that fit is (and isn't) safe to sample from for simulation, and turns it into a
menu of stress-test scenarios you can throw at a scheduling algorithm.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Fit distributions to a real (or your own) trace
gputrace analyze path/to/trace.csv --loader alibaba2020 --out fit_results.json

# 2. Generate one stress-test scenario
gputrace generate fit_results.json --scenario bursty_arrivals --n-jobs 20000 --out trace.csv

# 3. ...or every registered scenario at once, for a full benchmark suite
gputrace generate-all fit_results.json --n-jobs 20000 --out-dir synthetic_traces/

gputrace list-scenarios   # see what's available and what each one stresses
gputrace list-loaders     # see what data sources are supported
```

## Data sources supported out of the box

| loader name | source | notes |
|---|---|---|
| `alibaba2020` | [Alibaba cluster-trace-gpu-v2020](https://github.com/alibaba/clusterdata) | accepts either the official 100K-job simulator sample or the full `pai_task_table` release |
| `google2011` | [Google cluster trace (2011-2 schema)](https://github.com/google/cluster-data) | CPU-only (this trace predates public GPU accounting) — included as a differently-shaped reference workload, not a GPU data source |
| `synthetic` | any trace already in gputrace's unified schema | lets you re-analyze traces this framework generated |

**Adding a new source** is one file: subclass `gputrace.loaders.base.BaseLoader`,
implement `_load_raw()` and `_normalize()` to map your columns onto the unified schema
in `gputrace/schema.py`, and decorate the class with `@register("your_source_name")`.
Nothing else in the framework needs to change — see `loaders/alibaba2020.py` or
`loaders/google_cluster.py` for a template.

## Unified schema

Every loader outputs a DataFrame with the same columns
(`job_id, submit_time, duration, num_cpu, num_gpu, user, gpu_type, mem, wait_time,
status, source`) regardless of source, so analysis and generation code never needs to
know which dataset it's looking at. See `gputrace/schema.py`.

## Statistical analysis

`gputrace.analysis.analyze(df)` fits 8 candidate distributions (log-normal, Weibull,
gamma, Pareto, generalized Pareto, Burr12, log-logistic, exponential) via MLE to
`duration`, `num_cpu`, `num_gpu`, `wait_time`, and the inter-arrival-time process,
ranks them by AIC/BIC, reports the KS statistic (not just its p-value — with large
trace sizes, KS p-values are ~always significant even for a good fit; the *statistic*
is the informative part), and estimates the Hill tail index. It also segments
`duration` by `gpu_type` (or any categorical column) since pooling job classes
together washes out real heterogeneity.

**A deliberate safety check baked in:** whenever the best-AIC-fit distribution has a
shape parameter implying an infinite theoretical mean (Pareto with shape ≤ 1,
generalized Pareto with shape ≥ 1 — both common winners for bursty inter-arrival or
wait-time data), `analyze()` flags it via `infinite_mean_warning`, and the scenario
generator caps sampling from that distribution at a bounded multiple of the empirical
max, rather than silently producing simulated traces with unrealistic multi-day gaps
that never occurred in the source data. Best likelihood fit ≠ safe to extrapolate from.

## Stress-test scenarios

| scenario | stresses |
|---|---|
| `baseline` | calibration — reproduces the fitted trace as-is |
| `bursty_arrivals` | admission control, queue-depth spikes (Hawkes-style self-exciting arrivals) |
| `heavy_tail_demand` | fragmentation, large-job starvation (fatter resource-request tail) |
| `diurnal_pattern` | autoscaling / capacity planning across a day/night cycle |
| `high_contention` | the point where queueing/backlog collapses under scaled-up demand |
| `flash_crowd` | worst-case queueing from one extreme burst window |
| `resource_starvation` | whether small jobs starve behind large resource holders |
| `cold_start_storm` | scheduling overhead under a rapid-fire short-job storm |

**Adding a new scenario** is one function: write
`def my_scenario(gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame`
using the sampling/arrival-process primitives on `gen`, decorate it with
`@register_scenario("my_scenario", "what it stresses")` in
`generators/stress_scenarios.py` (or your own module that imports from it), and it's
immediately available via `gputrace generate --scenario my_scenario`.

## Development

```bash
pip install -e ".[dev]"
pytest                       # full test suite (43 tests: schema, loaders, distribution
                              # fitting incl. parameter-recovery, scenario generation
                              # incl. reproducibility, CLI end-to-end)
ruff check src tests         # lint
```

CI (`.github/workflows/ci.yml`) runs lint, the test suite across Python 3.10–3.12, and
an end-to-end CLI smoke test on every push/PR to `main`.

Tests run entirely against small bundled fixtures (`tests/fixtures/`) — CI does not
depend on downloading real trace archives, since those aren't reliably accessible from
network-restricted CI runners and shouldn't be a build dependency anyway. Point the
CLI at a real downloaded trace locally to reproduce the full-scale analysis.

## License

Apache-2.0 (see `LICENSE`). Note: this framework does not redistribute any Alibaba or
Google trace data — you must download those separately from their respective official
sources, subject to their own terms.
