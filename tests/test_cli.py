import json
import subprocess
import sys

import pandas as pd


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "gputrace.cli", *args],
                           capture_output=True, text=True, cwd=cwd)


def test_cli_list_loaders():
    r = run_cli("list-loaders")
    assert r.returncode == 0
    assert "alibaba2020" in r.stdout
    assert "google2011" in r.stdout


def test_cli_list_scenarios():
    r = run_cli("list-scenarios")
    assert r.returncode == 0
    assert "bursty_arrivals" in r.stdout
    assert "flash_crowd" in r.stdout


def test_cli_analyze_then_generate_end_to_end(alibaba_csv, tmp_path):
    fit_out = tmp_path / "fit_results.json"
    r1 = run_cli("analyze", alibaba_csv, "--loader", "alibaba2020", "--out", str(fit_out))
    assert r1.returncode == 0, r1.stderr
    assert fit_out.exists()
    data = json.loads(fit_out.read_text())
    assert "duration" in data

    trace_out = tmp_path / "trace.csv"
    r2 = run_cli("generate", str(fit_out), "--scenario", "bursty_arrivals",
                 "--n-jobs", "200", "--out", str(trace_out))
    assert r2.returncode == 0, r2.stderr
    assert trace_out.exists()
    df = pd.read_csv(trace_out)
    assert len(df) == 200


def test_cli_generate_all(alibaba_csv, tmp_path):
    fit_out = tmp_path / "fit_results.json"
    run_cli("analyze", alibaba_csv, "--out", str(fit_out))
    out_dir = tmp_path / "traces"
    r = run_cli("generate-all", str(fit_out), "--n-jobs", "100", "--out-dir", str(out_dir))
    assert r.returncode == 0, r.stderr
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) >= 8  # every registered built-in scenario
