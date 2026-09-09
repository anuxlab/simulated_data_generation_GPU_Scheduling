# Getting real trace data

This repo ships only small synthetic fixtures (`tests/fixtures/`) for CI — no real
trace data is redistributed here. To run the full analysis on real data:

## Alibaba cluster-trace-gpu-v2020

```bash
# full multi-table release (task-level; loader aggregates to job level)
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz
tar xzf pai_task_table.tar.gz
gputrace analyze pai_task_table.csv --loader alibaba2020 --out fit_results.json

# OR the smaller, already job-level simulator sample (fastest way to try things out)
git clone --depth 1 https://github.com/alibaba/clusterdata.git
gputrace analyze clusterdata/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_duration_estimate_100K.csv \
    --loader alibaba2020 --out fit_results.json
```

## Google cluster trace (2011-2 schema)

```bash
# See https://github.com/google/cluster-data for the full download instructions
# (BigQuery export or GCS bucket, gzipped/sharded CSVs). Concatenate the task_events
# shard(s) you want and point the loader at the result:
gputrace analyze task_events-000-of-500.csv --loader google2011 --out fit_results_google.json
```

## Then generate stress-test traces from either

```bash
gputrace generate-all fit_results.json --n-jobs 50000 --out-dir synthetic_traces/
```

## Adding another real source (Azure Public Dataset, Microsoft Philly, your own private trace, ...)

Write a new file in `src/gputrace/loaders/`, e.g. `azure.py`:

```python
from .base import BaseLoader, register

@register("azure")
class AzureLoader(BaseLoader):
    name = "azure"
    def _load_raw(self, path):
        ...  # pd.read_csv(path) or however your source is shaped
    def _normalize(self, raw):
        ...  # map your columns onto job_id, submit_time, duration, num_cpu, num_gpu, ...
        return mapped_df
```

Import it in `src/gputrace/loaders/__init__.py` (`from . import azure  # noqa: F401`)
and it's immediately available as `--loader azure` everywhere in the CLI, with no
other code changes.
