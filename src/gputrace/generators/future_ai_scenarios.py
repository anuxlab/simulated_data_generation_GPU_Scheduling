"""
Forward-looking stress scenarios reflecting how AI/GPU cloud workloads are evolving
past what a 2020-2023 training-job trace captures on its own. Every scenario below
cites the paper/report it's grounded in; where a mechanism is a reasonable
extrapolation rather than something directly measured in a public trace, that's
called out explicitly as an ASSUMPTION rather than presented as measured fact.

These still sample base job attributes from your fitted `fit_results` (so they stay
grounded in your real trace's duration/CPU/GPU distributions) but layer a new
*structural* mechanism on top -- bursty inference traffic, expert-load skew, spot
churn, etc. -- that a pure single-trace fit can't produce by itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scenario_generator import WorkloadScenarioGenerator
from .stress_scenarios import register_scenario


@register_scenario(
    "llm_inference_bursty",
    "Short, heavy-tailed-latency inference requests (not training jobs) with sharp "
    "bursty arrivals and a prefill/decode-style duration split. Evidence: Splitwise "
    "(Patel et al., ISCA 2024) and DynamoLLM (Stojkovic et al. 2025, Microsoft) both "
    "document that bursty request rates cause prefill queues to saturate while decode "
    "GPUs sit idle; DynaServe's analysis of the Azure Code and BurstGPT production "
    "traces shows persistent, rapidly-alternating prefill-heavy and decode-heavy "
    "periods, i.e. high temporal variance is a measured production property, not a "
    "hypothesis. Stress-tests: prefill/decode-style scheduling, autoscaling reaction time.",
)
def llm_inference_bursty(gen: WorkloadScenarioGenerator, n_jobs: int,
                          burst_prob: float = 0.5, burst_size_range=(10, 80),
                          duration_scale: float = 0.02, **_) -> pd.DataFrame:
    # ASSUMPTION: we approximate "inference request" duration as a heavily compressed
    # version of the fitted training-job duration (requests are seconds, not hours) --
    # the *shape* (heavy right tail from occasional long generations) is kept from the
    # real fit, only the scale is assumed.
    submit_time = gen.arrivals_bursty_hawkes(n_jobs, burst_prob=burst_prob,
                                              burst_size_range=burst_size_range, burst_width=2.0)
    duration = gen.sample_metric("duration", n_jobs) * duration_scale
    # ASSUMPTION: num_cpu is repurposed as a proxy for prompt+output token count
    # (heavy-tailed, as documented for prompt-length distributions in the papers above).
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = np.clip(gen.sample_metric("num_gpu", n_jobs), 0, 1)  # inference: fractional/1 GPU typical
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "llm_inference_bursty")


@register_scenario(
    "long_context_kv_pressure",
    "Per-job memory/context footprint (proxied via num_cpu) trends upward across the "
    "trace window, with an increasing rate of extreme outliers. Evidence: Alibaba's "
    "Infinite-LLM system report states production context lengths range from a few "
    "tokens to more than two million, cited via DynaServe's analysis -- i.e. the extreme "
    "tail this scenario stresses is a real observed production value, not a projection. "
    "ASSUMPTION: the *growth-over-time* mechanism (context lengths creeping up across "
    "the trace window) is our extrapolation of the well-documented industry trend toward "
    "longer context windows generation over generation -- we are not aware of a public "
    "trace that measures this within-trace drift directly. Stress-tests: memory "
    "fragmentation and OOM risk as average job footprint grows, KV-cache-aware placement.",
)
def long_context_kv_pressure(gen: WorkloadScenarioGenerator, n_jobs: int,
                              start_mult: float = 1.0, end_mult: float = 4.0,
                              extreme_prob: float = 0.01, extreme_mult: float = 40.0,
                              **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    trend = gen.time_trend_scale(submit_time, start_mult, end_mult)
    num_cpu = gen.sample_metric("num_cpu", n_jobs) * trend
    is_extreme = gen.rng.random(n_jobs) < extreme_prob
    num_cpu = np.where(is_extreme, num_cpu * extreme_mult, num_cpu)  # the "2M-token" tail
    num_gpu = gen.sample_metric("num_gpu", n_jobs) * np.clip(trend, 1, None)
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "long_context_kv_pressure")


@register_scenario(
    "spot_preemption_churn",
    "A fraction of jobs are preempted mid-run (spot/preemptible capacity reclaimed) "
    "and resubmitted as a follow-up attempt with restart/checkpoint-reload overhead "
    "added to their effective duration. Evidence: Lazarus (arXiv 2407.04656) reports "
    "LLM training failure rates as high as 44% and spot preemptions occurring as "
    "frequently as every 5-10 minutes; PCcheck (ASPLOS'25) instruments actual "
    "preemption traces from a 64-A100 spot cluster on GCP over a 16-hour window. "
    "ASSUMPTION: the exact preemption-rate default below (18%) and recovery-overhead "
    "multiplier (1.15-1.6x) are our estimates informed by, but not lifted verbatim "
    "from, those sources -- tune `preempt_rate` to match a rate you've measured or "
    "been quoted by your cloud provider. Stress-tests: checkpoint/restart-aware "
    "scheduling, whether a policy wastes cluster time on jobs that get killed anyway.",
)
def spot_preemption_churn(gen: WorkloadScenarioGenerator, n_jobs: int,
                           preempt_rate: float = 0.18,
                           recovery_overhead_range=(1.15, 1.6), **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    preempted = gen.rng.random(n_jobs) < preempt_rate
    overhead = gen.rng.uniform(*recovery_overhead_range, size=n_jobs)
    duration = np.where(preempted, duration * overhead, duration)
    df = gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "spot_preemption_churn")
    df["status"] = np.where(preempted, "PREEMPTED_RESTARTED", "COMPLETED")
    return df


@register_scenario(
    "checkpoint_io_burst",
    "Periodic, synchronized bursts of short, high-CPU/memory 'checkpoint write' "
    "pseudo-jobs injected across the whole cluster at a fixed interval, overlapping "
    "normal traffic. Evidence: PCcheck, CheckFreq, and Gemini (all cited in PCcheck, "
    "ASPLOS'25) are entire systems built around the measured cost of periodic "
    "checkpoint I/O for large-model training; industry guidance (spot-training blog "
    "posts, ASSUMPTION-flagged as lower-rigor than the peer-reviewed sources above) "
    "converges on checkpointing every 500-1000 steps, taking tens of seconds per "
    "checkpoint even for a single large model shard -- at cluster scale with many "
    "concurrent training jobs, many such checkpoints landing in the same window is a "
    "reasonable inference from that guidance, though we are not aware of a public "
    "multi-job trace that measures cluster-wide checkpoint synchronization directly, "
    "so treat the *synchronization* aspect specifically as an ASSUMPTION. "
    "Stress-tests: network/storage contention windows, whether scheduling naively "
    "co-locates checkpoint I/O with latency-sensitive jobs.",
)
def checkpoint_io_burst(gen: WorkloadScenarioGenerator, n_jobs: int,
                         checkpoint_period_s: float = 1800.0,
                         checkpoint_fraction: float = 0.15,
                         checkpoint_duration_s: float = 40.0, **_) -> pd.DataFrame:
    n_ckpt = max(int(n_jobs * checkpoint_fraction), 1)
    n_base = n_jobs - n_ckpt
    base_times = gen.arrivals_renewal(n_base)
    span = base_times.max() - base_times.min() if n_base > 0 else checkpoint_period_s * n_ckpt
    ckpt_times = gen.periodic_times(checkpoint_period_s, n_ckpt, jitter_frac=0.05) % max(span, 1.0)

    base_duration = gen.sample_metric("duration", n_base) if n_base > 0 else np.array([])
    base_cpu = gen.sample_metric("num_cpu", n_base) if n_base > 0 else np.array([])
    base_gpu = gen.sample_metric("num_gpu", n_base) if n_base > 0 else np.array([])
    base_gtype = gen.sample_gpu_type(n_base) if n_base > 0 else np.array([])

    ckpt_duration = np.full(n_ckpt, checkpoint_duration_s) * gen.rng.uniform(0.7, 1.3, size=n_ckpt)
    ckpt_cpu = gen.sample_metric("num_cpu", n_ckpt) * 3.0  # I/O-heavy write burst
    ckpt_gpu = np.zeros(n_ckpt)  # checkpoint write itself doesn't hold the GPU
    ckpt_gtype = np.full(n_ckpt, "CPU")

    submit_time = np.concatenate([base_times, ckpt_times])
    duration = np.concatenate([base_duration, ckpt_duration])
    num_cpu = np.concatenate([base_cpu, ckpt_cpu])
    num_gpu = np.concatenate([base_gpu, ckpt_gpu])
    gpu_type = np.concatenate([base_gtype, ckpt_gtype])
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "checkpoint_io_burst")


@register_scenario(
    "gpu_fragmentation_sharing",
    "Many jobs request small fractional-GPU slices across a wide mix of GPU types, "
    "creating bin-packing fragmentation. Evidence: Alibaba's own follow-up trace and "
    "paper, 'Beware of Fragmentation: Scheduling GPU-Sharing Workloads with "
    "Fragmentation Gradient Descent' (Weng et al., USENIX ATC 2023), is built entirely "
    "around this measured problem in their cluster-trace-gpu-v2023 release (6,200+ "
    "GPUs across ~1,200 machines with diverse, heterogeneous per-node GPU "
    "configurations). Stress-tests: bin-packing quality, whether a scheduler's "
    "placement policy leaves stranded fractional-GPU capacity.",
)
def gpu_fragmentation_sharing(gen: WorkloadScenarioGenerator, n_jobs: int,
                               gpu_fractions=(0.125, 0.25, 0.5, 1.0), **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    # ASSUMPTION: fraction weights (more small slices than whole GPUs) are chosen to
    # produce a fragmentation-heavy mix; tune to your own fleet's sharing granularity.
    frac_weights = np.array([0.35, 0.30, 0.20, 0.15])
    num_gpu = gen.rng.choice(gpu_fractions, size=n_jobs, p=frac_weights)
    gpu_type = gen.sample_gpu_type(n_jobs)
    num_gpu = np.where(gpu_type == "CPU", 0.0, num_gpu)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "gpu_fragmentation_sharing")


@register_scenario(
    "moe_expert_load_skew",
    "Jobs are tagged to one of N 'expert' shards (encoded in the `user` field as "
    "expert_<i>) with Zipf-skewed popularity, so a handful of experts/tenants receive "
    "disproportionate load while others sit near-idle -- the MoE 'straggler' pattern. "
    "Evidence: measured directly in production-scale models -- for Qwen3-235B (128 "
    "experts), the most-used expert is invoked 2.4x more often than a uniform "
    "distribution would predict (GEM, 2026); for Mixtral-8x7B, two GPUs were found to "
    "process 64% and 69% of tokens in specific layers respectively (MoETuner, 2025); "
    "MoEless replays real Azure LLM inference traces to show this skew directly drives "
    "GPU straggler effects in production-shaped traffic. Stress-tests: load-aware "
    "placement, whether a policy detects and rebalances persistent hot-spot load "
    "rather than just balancing job *count* per GPU.",
)
def moe_expert_load_skew(gen: WorkloadScenarioGenerator, n_jobs: int,
                          n_experts: int = 32, skew: float = 1.2, **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    weights = gen.zipf_weights(n_experts, skew=skew)
    expert_id = gen.rng.choice(n_experts, size=n_jobs, p=weights)
    # hot experts get proportionally more load per job, not just more jobs --
    # compounding count-skew with per-job-size skew, matching the "GPUs hosting hot
    # experts take longer" mechanism described in the cited papers.
    hotness = weights[expert_id] / weights.max()
    num_cpu = gen.sample_metric("num_cpu", n_jobs) * (0.5 + 1.5 * hotness)
    num_gpu = gen.sample_metric("num_gpu", n_jobs) * (0.5 + 1.5 * hotness)
    gpu_type = gen.sample_gpu_type(n_jobs)
    df = gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "moe_expert_load_skew")
    df["user"] = [f"expert_{i}" for i in expert_id]
    return df
