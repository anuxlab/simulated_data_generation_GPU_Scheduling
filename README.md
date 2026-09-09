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
| `llm_inference_bursty` | prefill/decode-style scheduling under bursty inference traffic |
| `long_context_kv_pressure` | memory fragmentation / OOM risk as average job footprint grows over time |
| `spot_preemption_churn` | checkpoint/restart-aware scheduling under spot-capacity reclamation |
| `checkpoint_io_burst` | network/storage contention from synchronized periodic checkpoint I/O |
| `gpu_fragmentation_sharing` | bin-packing quality under fractional-GPU, heterogeneous-type demand |
| `moe_expert_load_skew` | load-aware placement under Zipf-skewed per-tenant/expert demand |

See **"Related work / evidence for each scenario"** below for citations and explicit
assumption flags — several of the AI-era scenarios mix a measured finding (e.g. the
magnitude of MoE expert skew) with an assumed mechanism (e.g. how that skew evolves
over a trace window), and the table is explicit about which is which.

**Adding a new scenario** is one function: write
`def my_scenario(gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame`
using the sampling/arrival-process primitives on `gen`, decorate it with
`@register_scenario("my_scenario", "what it stresses")` in
`generators/stress_scenarios.py` (or your own module that imports from it), and it's
immediately available via `gputrace generate --scenario my_scenario`.

## Related work / evidence for each scenario

Every scenario below is grounded either in a specific measurement from a peer-reviewed
paper or public trace, or explicitly flagged as an **ASSUMPTION** where we're
extrapolating rather than citing a direct measurement. None of this is "trust me" —
check the sources.

### Core scenarios (present-day cluster/cloud trace patterns)

| Scenario | Evidence |
|---|---|
| `baseline` | This is the source paper for the exact dataset the framework ships fixtures for: a characterization of a two-month production MLaaS trace from a 6,000+ GPU Alibaba cluster, covering low GPU utilization and long queueing delays (Weng et al., ["MLaaS in the Wild"](https://www.usenix.org/conference/nsdi22/presentation/weng), NSDI '22). |
| `bursty_arrivals` | Directly measured in our own analysis: CV=1.80 for inter-arrival times and 19.4% of jobs landing in the same second (see `REPORT.md`). Corroborated in the literature: Google's cluster trace shows large spikes and troughs in concurrent task counts over time, and foundational trace-analysis work explicitly characterizes cluster workloads as highly dynamic and driven by many short jobs demanding fast scheduling decisions (Reiss et al., [SoCC 2012](https://www.usenix.org/conference/nsdi22/presentation/weng); [CloudCoaster, arXiv:1907.02162](https://arxiv.org/pdf/1907.02162)). |
| `heavy_tail_demand` | One of the best-established findings in the field: cloud resource-usage distributions have tails heavier than log-normal, exponential, or normal, undermining averaging-based capacity models ([Journal of Grid Computing, 2012](https://link.springer.com/article/10.1007/s10723-012-9211-x)). Heavy-tailed job sizes are documented across web files, process lifetimes, and datacenter jobs, and provably require different scheduling policies than light-tailed workloads ([SPLIT, arXiv:2605.13749](https://arxiv.org/pdf/2605.13749)). |
| `diurnal_pattern` | Production Azure Functions traces show peak-period arrival rates roughly double the daily average (Shahrad et al., cited via [LACE](https://link.springer.com/article/10.1007/s44443-026-00473-4)). Cortez et al.'s ["Resource Central"](https://www.microsoft.com/en-us/research/publication/resource-central-understanding-predicting-workloads-improved-resource-management-large-cloud-platforms/) (SOSP '17) is the standard reference for temporal VM-workload characterization at Azure scale. |
| `flash_crowd` | The seminal characterization of sudden legitimate-traffic surges that overload services (Jung, Krishnamurthy & Rabinovich, ["Flash Crowds and Denial of Service Attacks"](https://dl.acm.org/doi/10.1145/511446.511485), WWW 2002 — 800+ citations). |
| `resource_starvation` | The well-documented head-of-line-blocking problem: Sparrow's decentralized scheduler faces challenges under heavy load from a lack of head-of-line-blocking mitigation, motivating follow-up schedulers Hawk and Eagle built specifically to stop large jobs blocking short ones ([Sparrow, SOSP '13](https://people.eecs.berkeley.edu/~matei/papers/2013/sosp_sparrow.pdf); [Peacock, arXiv:1805.04449](https://arxiv.org/pdf/1805.04449)). |
| `cold_start_storm` | Azure Functions production traces show most functions are invoked very infrequently but span an 8-order-of-magnitude range of invocation frequencies — exactly the bursts-after-idle pattern that drives the serverless cold-start problem ([Shahrad et al., ATC 2020](https://www.usenix.org/conference/atc20/presentation/shahrad)). |
| `high_contention` | Not tied to one paper — a standard load-testing technique (turn up demand until something breaks), motivated by the same capacity-planning fragility heavy-tailed usage creates operationally. **Treat as a stress-testing convention, not a directly-measured pattern.** |

### AI/LLM-era scenarios (where cloud/GPU workloads are heading)

These extend the framework past what a 2020-era training-job trace captures on its
own — current production LLM serving and training infrastructure has moved a long way
since then. Docstrings in `generators/future_ai_scenarios.py` carry the full citation
and assumption flags inline; summary below.

| Scenario | Evidence | What's measured vs. assumed |
|---|---|---|
| `llm_inference_bursty` | Splitwise ([Patel et al., ISCA 2024](https://arxiv.org/html/2311.18677v2)) and DynamoLLM (Stojkovic et al. 2025, Microsoft) document that bursty request rates saturate prefill queues while decode GPUs idle; DynaServe's analysis of the real Azure Code and BurstGPT production traces shows persistent, rapidly-alternating prefill-/decode-heavy periods ([arXiv:2504.09285](https://arxiv.org/pdf/2504.09285)). | High temporal variance in real serving traces: **measured**. Compressing our fitted training-job duration into a request-latency proxy: **assumption** (shape kept, scale assumed). |
| `long_context_kv_pressure` | Alibaba's Infinite-LLM system reports production context lengths ranging from a few tokens to over two million, per DynaServe's citation of it ([arXiv:2504.09285](https://arxiv.org/pdf/2504.09285)). | The 2M-token extreme tail: **measured, real production value**. The within-trace *growth-over-time* mechanism: **assumption** — our extrapolation of the well-documented cross-generation trend toward longer context windows; we don't know of a public trace measuring this drift within a single trace window. |
| `spot_preemption_churn` | Lazarus reports LLM training failure rates as high as 44% and spot preemptions occurring as often as every 5-10 minutes ([arXiv:2407.04656](https://arxiv.org/pdf/2407.04656)); PCcheck instruments real preemption traces from a 64-A100 spot cluster on GCP over 16 hours ([ASPLOS '25](https://anakli.inf.ethz.ch/papers/PCcheck_asplos25.pdf)). | Preemption frequency and failure-rate *order of magnitude*: **measured**. Our specific default preemption rate (18%) and recovery-overhead multiplier (1.15-1.6x): **assumption**, informed by but not lifted from these numbers — tune to your own provider's quoted rates. |
| `checkpoint_io_burst` | PCcheck, CheckFreq, and Gemini are entire systems built around the measured cost of periodic large-model checkpoint I/O (all cited in PCcheck, ASPLOS '25). | The *cost* of individual checkpoints: **measured** (peer-reviewed systems papers). The *cluster-wide synchronization* of many concurrent jobs' checkpoints landing in the same window: **assumption** — a reasonable inference from per-job checkpoint-interval guidance, but we don't know of a public multi-job trace measuring this directly. Industry blog guidance (checkpoint every 500-1000 steps) is noted in the docstring as lower-rigor than the peer-reviewed sources. |
| `gpu_fragmentation_sharing` | Alibaba's own "Beware of Fragmentation" paper (Weng et al., [USENIX ATC 2023](https://www.usenix.org/conference/atc23/presentation/weng)) and its accompanying `cluster-trace-gpu-v2023` release (6,200+ GPUs across ~1,200 machines with diverse heterogeneous configurations) are built entirely around this measured problem. | **Measured** — this is the closest of all scenarios to a directly-cited production trace finding. |
| `moe_expert_load_skew` | Measured directly in production-scale models: for Qwen3-235B (128 experts) the hottest expert is invoked 2.4x more than a uniform distribution predicts ([GEM, 2026](https://arxiv.org/html/2605.19945)); for Mixtral-8x7B, two GPUs were found processing 64% and 69% of tokens in specific layers ([MoETuner, arXiv:2502.06643](https://arxiv.org/pdf/2502.06643)); MoEless replays real Azure LLM inference traces to show this skew directly drives GPU straggler effects in production traffic ([arXiv:2603.06350](https://arxiv.org/pdf/2603.06350)). | Expert-popularity skew magnitude: **measured**, multiple independent sources. The Zipf-distribution *shape* used to generate it and the specific `skew` default: **assumption** — a standard way to parameterize power-law popularity, not itself measured for this exact use case. |

If you're citing this framework's scenario design in a paper or internal doc, cite the
underlying sources above directly rather than this repo — we're aggregating and
translating their findings into synthetic-trace mechanics, not presenting original
measurements ourselves (except for the `bursty_arrivals` CV/simultaneous-arrival
numbers, which came from our own analysis of the bundled trace — see `REPORT.md`).

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
