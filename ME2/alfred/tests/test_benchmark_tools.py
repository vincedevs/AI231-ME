from __future__ import annotations

import pytest

from tools.benchmark_device import percentile, summarize
from tools.compare_benchmarks import comparison_rows, compatibility_warnings


def report(label: str, p50: float, p95: float, workload: str = "same") -> dict:
    return {
        "benchmark_version": "1.0",
        "label": label,
        "parameters": {"warmup": 5, "runs": 30},
        "workload": {"sha256": workload},
        "models": {"manifest_sha256": "manifest"},
        "system": {"peak_rss_mib_after": 100.0},
        "stages": {
            "vcm": {
                "initialization_ms": 5.0,
                "output": {"decision": "execute", "intent": "TIME"},
                "steady_state": {
                    "summary": {
                        "p50_ms": p50,
                        "p95_ms": p95,
                        "real_time_factor_p95": p95 / 3000,
                    }
                },
            }
        },
    }


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_summary_reports_latency_and_real_time_factor() -> None:
    result = summarize([10.0, 20.0, 30.0], audio_seconds=2.0)
    assert result["runs"] == 3
    assert result["mean_ms"] == 20
    assert result["p50_ms"] == 20
    assert result["real_time_factor_p50"] == pytest.approx(0.01)


def test_comparison_reports_candidate_slowdown_ratio() -> None:
    baseline = report("macbook", 10, 12)
    candidate = report("raspberry_pi_5", 20, 30)
    rows = comparison_rows(baseline, candidate)
    p50 = next(row for row in rows if row["metric"] == "p50_ms")
    assert p50["candidate_over_baseline"] == 2.0
    assert compatibility_warnings(baseline, candidate) == []


def test_comparison_includes_initialization_latency() -> None:
    rows = comparison_rows(report("macbook", 10, 12), report("raspberry_pi_5", 20, 30))
    initialization = next(row for row in rows if row["metric"] == "initialization_ms")
    assert initialization["baseline"] == 5.0
    assert initialization["candidate"] == 5.0


def test_comparison_warns_about_nonidentical_workloads() -> None:
    warnings = compatibility_warnings(
        report("macbook", 10, 12, workload="first"),
        report("raspberry_pi_5", 20, 30, workload="second"),
    )
    assert warnings == ["Different workload waveform: 'first' versus 'second'"]


def test_comparison_warns_when_model_outputs_differ() -> None:
    baseline = report("macbook", 10, 12)
    candidate = report("raspberry_pi_5", 20, 30)
    candidate["stages"]["vcm"]["output"]["intent"] = "WEATHER"
    assert compatibility_warnings(baseline, candidate) == ["VCM outputs differ across devices"]
