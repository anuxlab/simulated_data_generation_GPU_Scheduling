# gputrace

Synthetic GPU-cluster workload trace generation for benchmarking schedulers
and cluster simulators. Fits distributions to a real cluster trace, then
generates arbitrarily large synthetic traces under one of 14 stress
scenarios that stay faithful to your source data's marginal distributions
while injecting a specific, named form of workload stress (bursty arrivals,
GPU-type contention, fractional-GPU fragmentation, spot preemption churn,
etc).

Built to answer one specific problem: a single baseline trace (or its
sample) is not a diverse enough test set to know whether a scheduling
policy is actually robust — it's one point in a huge space of possible
arrival dynamics and demand mixes. This framework lets you hold the
demand *distribution* fixed (so you're not confounding "different
scheduler" with "different workload") while varying the *dynamics* along
14 axes that matter in practice.

## Install

```bash
pip install -e .
```

Requires Python 3.10+, pandas, numpy, scipy.

## Quickstart

```bash
# 1. Fit distributions to a real trace (Alibaba cluster-trace-gpu-v2020 sample)
gputrace analyze --loader alibaba2020 --input pai_task_table_sample.csv --out fit.json

# 2. Generate one scenario
gputrace generate --scenario bursty_arrivals --fit fit.json --n-jobs 50000 --seed 1 --out bursty.csv

# 3. ...or generate every registered scenario at once
gputrace generate-all --fit fit.json --n-jobs 20000 --seed 1 --out-dir traces/

# 4. Export a scenario into a scheduler simulator's native input format
gputrace export --input bursty.csv --format k8s_sim --out-dir k8s_input/ --n-nodes 20 --gpus-per-node 8
```

No real trace on hand? Every command works with a built-in reference fit —
omit `--fit` on `generate`/`generate-all`, or call `gputrace.reference_fit()`
from Python. This is a smoke-test fixture for demos/CI, not a substitute
for fitting your own production trace (see docstring in
`generators/_reference.py` for why).

### Python API

```python
import gputrace as gt

fit = gt.analyze_file("alibaba2020", "pai_task_table_sample.csv")
df = gt.generate("high_contention", fit, n_jobs=10_000, seed=1, hot_type="A100")
df.to_csv("high_contention.csv", index=False)
```

## The unified schema

Every loader and every scenario generator produces the same column set —
this is what makes new data sources and new scenarios drop-in compatible
with everything downstream:

| column | type | meaning |
|---|---|---|
| `job_id` | str | unique within the trace |
| `submit_time` | float | seconds since trace start |
| `duration` | float | seconds of actual runtime (excludes queueing) |
| `num_cpu` | float | vCPU cores requested |
| `num_gpu` | float | GPU devices requested (whole count) |
| `user` | str | tenant / submitting user |
| `gpu_type` | str | requested SKU, e.g. `A100`, `V100`, `T4`, `H100`, or `""` |
| `mem` | float | GiB requested |
| `wait_time` | float | queueing delay (0 for generated scenarios — that's the simulator's output, not the generator's input) |
| `status` | str | `completed` / `failed` / `killed` |
| `source` | str | which loader/scenario produced the row |
| `gpu_milli` *(optional)* | float | fractional-GPU request in milli-units, 0-1000 per device. Populated only by scenarios that model GPU sharing (`gpu_fragmentation_sharing`, `llm_inference_bursty`); absent elsewhere, in which case consumers should assume 1000 (whole device) per requested GPU. |

Full column semantics and the fractional-GPU convention are documented in
`src/gputrace/schema.py`. `schema.validate(df)` checks any DataFrame
against this contract — every loader and generator runs it before
returning.

## Loaders (input side)

Built in: `alibaba2020` (cluster-trace-gpu-v2020), `google2011` (2011-2
schema, CPU-only reference workload), `synthetic` (passthrough for
re-analyzing gputrace's own output). Add a new source in one file:

```python
from gputrace.loaders.base import BaseLoader, register

@register("my_source")
class MyLoader(BaseLoader):
    def _load_raw(self, path, **kwargs):
        ...  # read into whatever shape is natural
    def _normalize(self, raw):
        ...  # return a DataFrame with the unified schema's columns
```

`gputrace list-loaders` lists what's registered.

## Scenarios (output side)

14 built in, each documented with exactly what it stresses:

```
$ gputrace list-scenarios
baseline                     reference: homogeneous Poisson arrivals, fitted marginals, no injected stress
bursty_arrivals              self-exciting (Hawkes) arrivals: clustered bursts, CV(inter-arrival) > 1
checkpoint_io_burst          periodic synchronized duration spikes, modeling checkpoint/IO stalls
cold_start_storm             long idle gaps punctuated by rapid-fire short-job storms
diurnal_pattern               day/night sinusoidal load cycle
flash_crowd                  one concentrated arrival spike against a low background rate
gpu_fragmentation_sharing    fractional-GPU requests (gpu_milli populated) stressing bin-packing
heavy_tail_demand            resource-request sizes sampled with the infinite-mean cap raised
high_contention               many jobs concentrated on one popular GPU SKU
llm_inference_bursty         short high-QPS inference-style jobs, Hawkes arrivals, small footprints
long_context_kv_pressure     subset of jobs with very long duration + elevated memory
moe_expert_load_skew         Zipf-skewed per-tenant demand (hot experts/tenants)
resource_starvation           a small set of users submit disproportionately large jobs
spot_preemption_churn        short-lived jobs, high kill rate (spot/preemptible instances)
```

All scenarios share the same fitted marginal distributions for job size
(duration, CPU/GPU demand) — what differs is the *arrival process* and, for
a few, resource-request *correlation structure*. This is deliberate: it
isolates "does this scheduler handle bursty dynamics" from "does it handle
a different size mix," which would otherwise be confounded.

Add a scenario in one function:

```python
from gputrace.generators.engine import register_scenario, WorkloadScenarioGenerator

@register_scenario("my_scenario", "what it stresses, one line")
def my_scenario(gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame:
    submit_times = gen.poisson_arrivals(n_jobs, rate=1.0)
    ...
```

`WorkloadScenarioGenerator` (`generators/engine.py`) exposes the sampling
primitives: `sample_field` (bootstrap for low-cardinality fields like
`num_gpu`, parametric MLE-fit sampling otherwise, with an infinite-mean
safety cap — see below), `poisson_arrivals`, `hawkes_arrivals`,
`diurnal_arrivals`, `flash_crowd_arrivals`, `cold_start_arrivals`, and
`zipf_weights`.

## Two fitting safety nets (read this before trusting fit quality)

1. **Low-cardinality fields bootstrap instead of fitting a curve.**
   `num_gpu` and similar small-integer fields (a handful of unique values
   like `{0,1,2,4,8}`) break continuous MLE fitting — the optimizer can
   drive a shape parameter to a degenerate extreme that technically
   maximizes likelihood on repeated values but explodes when sampled from.
   `analysis.py` detects this (≤25 unique values, or <2% unique/n) and
   switches to empirical bootstrap resampling automatically. Check
   `FieldFit.low_cardinality` to see which fields this applied to.

2. **Infinite-mean fits are capped, not silently sampled raw.** Pareto
   (shape ≤ 1) and generalized-Pareto (shape ≥ 1) fits are common winners
   for bursty inter-arrival or wait-time data and imply an *infinite*
   theoretical mean. Best-AIC does not mean safe-to-extrapolate-from:
   sampling from a genuinely infinite-mean fit produces synthetic traces
   with unrealistic multi-day gaps that never occurred in the source data.
   `WorkloadScenarioGenerator.sample_field` caps such samples at
   `cap_multiple × empirical_max` (default 5×; `heavy_tail_demand` raises
   this deliberately to 20× since exposing tail behavior is its point).
   Check `FieldFit.infinite_mean_warning`.

## Exporting into a simulator (py_sim / k8s_sim)

`gputrace export --format k8s_sim` writes `pods.csv` + `nodes.csv` in the
schema `py_sim`'s `k8s_sim.gputrace_bridge` module consumes — see that
package's README for the other half of this bridge, including why a
snapshot scheduler needs a time-driven wrapper to actually exercise the
scenarios that are about *arrival dynamics* rather than just size mix.

## Analysis output (`fit.json`)

`gputrace analyze` writes an `AnalysisResult`: per-field distribution
rankings (AIC/BIC/KS statistic), the Hill tail-index estimate, and
`duration` fits segmented by `gpu_type`. Load it back with
`gputrace.AnalysisResult.from_json(path)`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

31 tests covering schema validation, every scenario's schema conformance,
seed-reproducibility, the Hawkes-vs-Poisson clustering sanity check, the
infinite-mean cap, and the k8s_sim exporter.

## Roadmap

See the parent conversation / `py_sim/README.md` "Advanced experiments"
section — planned: a cluster/topology generator (currently only the
*demand* side is generated, exported cluster shapes are heuristic), a
streaming/generator Python API instead of DataFrame materialization, and
richer fractional-GPU modeling informed by real GPU-sharing telemetry
rather than the current fixed-quanta heuristic.
