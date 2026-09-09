# Contributing

## Adding a new dataset loader

1. Create `src/gputrace/loaders/<name>.py`, subclassing `BaseLoader`
   (`src/gputrace/loaders/base.py`):
   - `_load_raw(path)` — read your source file(s) into a DataFrame with native columns.
   - `_normalize(raw)` — map those columns onto the unified schema
     (`src/gputrace/schema.py`): `job_id, submit_time, duration, num_cpu, num_gpu`
     required; `user, gpu_type, mem, wait_time, status` optional (defaults are filled
     in automatically by `schema.validate`).
   - Decorate the class with `@register("<name>")`.
2. Import the module in `src/gputrace/loaders/__init__.py`.
3. Add a small synthetic fixture under `tests/fixtures/` shaped like your real source
   (don't commit real trace data or anything large) and a test in `tests/test_loaders.py`
   asserting the normalized output satisfies the unified schema.
4. Document any source-specific caveats in the loader's docstring (see
   `google_cluster.py` for the style — e.g. this source has no GPU data, or these
   fields are pre-normalized to [0,1]).

## Adding a new stress-test scenario

1. In `src/gputrace/generators/stress_scenarios.py` (or your own module), write:
   ```python
   @register_scenario("my_scenario", "one-line description of what this stresses")
   def my_scenario(gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame:
       submit_time = gen.arrivals_renewal(n_jobs)   # or arrivals_bursty_hawkes / arrivals_diurnal
       duration = gen.sample_metric("duration", n_jobs)
       num_cpu = gen.sample_metric("num_cpu", n_jobs)
       num_gpu = gen.sample_metric("num_gpu", n_jobs)
       gpu_type = gen.sample_gpu_type(n_jobs)
       return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "my_scenario")
   ```
2. Add it to `EXPECTED_SCENARIOS` in `tests/test_generators.py` so the standard suite
   (schema validity, reproducibility with a fixed seed) covers it automatically.
3. If it needs a genuinely new sampling primitive (not just a different combination of
   existing ones), add it as a method on `WorkloadScenarioGenerator` in
   `scenario_generator.py`, not inline in the scenario function, so other scenarios can
   reuse it.

## Running checks locally before opening a PR

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

Both must pass — CI runs the same two commands (plus a Python-version matrix and an
end-to-end CLI smoke test) on every push/PR to `main`.
