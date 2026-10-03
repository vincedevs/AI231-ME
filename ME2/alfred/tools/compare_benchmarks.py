#!/usr/bin/env python3
"""Compare two Alfred hardware benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if report.get("benchmark_version") != "1.0":
        raise ValueError(f"Unsupported benchmark report: {path}")
    return report


def compatibility_warnings(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    warnings = []
    baseline_workload = baseline["workload"]
    candidate_workload = candidate["workload"]
    checks = (
        (
            "workload waveform",
            baseline_workload.get("waveform_sha256") or baseline_workload.get("sha256"),
            candidate_workload.get("waveform_sha256") or candidate_workload.get("sha256"),
        ),
        (
            "model manifest",
            baseline["models"].get("manifest_sha256"),
            candidate["models"].get("manifest_sha256"),
        ),
        (
            "warm-up count",
            baseline["parameters"].get("warmup"),
            candidate["parameters"].get("warmup"),
        ),
        (
            "run count",
            baseline["parameters"].get("runs"),
            candidate["parameters"].get("runs"),
        ),
    )
    for label, first, second in checks:
        if first != second:
            warnings.append(f"Different {label}: {first!r} versus {second!r}")
    baseline_vcm = baseline["stages"].get("vcm", {}).get("output", {})
    candidate_vcm = candidate["stages"].get("vcm", {}).get("output", {})
    output_fields = ("decision", "intent", "slots")
    baseline_signature = tuple(baseline_vcm.get(field) for field in output_fields)
    candidate_signature = tuple(candidate_vcm.get(field) for field in output_fields)
    if baseline_signature != candidate_signature:
        warnings.append("VCM outputs differ across devices")
    baseline_asr = baseline["stages"].get("moonshine_asr", {}).get("transcript")
    candidate_asr = candidate["stages"].get("moonshine_asr", {}).get("transcript")
    if baseline_asr != candidate_asr:
        warnings.append("Moonshine transcripts differ across devices")
    return warnings


def comparison_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shared = sorted(set(baseline["stages"]) & set(candidate["stages"]))
    for stage in shared:
        for metric in ("initialization_ms", "cold_start_ms", "cold_start_inference_ms"):
            if metric not in baseline["stages"][stage] or metric not in candidate["stages"][stage]:
                continue
            baseline_value = float(baseline["stages"][stage][metric])
            candidate_value = float(candidate["stages"][stage][metric])
            rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "candidate_over_baseline": (
                        candidate_value / baseline_value if baseline_value else None
                    ),
                }
            )
        first = baseline["stages"][stage].get("steady_state", {}).get("summary")
        second = candidate["stages"][stage].get("steady_state", {}).get("summary")
        if not first or not second:
            continue
        for metric in ("p50_ms", "p95_ms", "real_time_factor_p95"):
            if metric not in first or metric not in second:
                continue
            baseline_value = float(first[metric])
            candidate_value = float(second[metric])
            rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "candidate_over_baseline": (
                        candidate_value / baseline_value if baseline_value else None
                    ),
                }
            )
    return rows


def markdown_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    lines = [
        "# Alfred hardware benchmark comparison",
        "",
        f"Baseline: **{baseline['label']}**  ",
        f"Candidate: **{candidate['label']}**",
        "",
        "A latency ratio above 1.00 means the candidate is slower; below 1.00 means it is faster.",
        "",
    ]
    if warnings:
        lines.extend(["## Comparability warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## Initialization and inference",
            "",
            "| Stage | Metric | Baseline | Candidate | Candidate / baseline |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ratio = row["candidate_over_baseline"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}×"
        lines.append(
            f"| {row['stage']} | {row['metric']} | {row['baseline']:.4f} | "
            f"{row['candidate']:.4f} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "## Whole-process peak memory",
            "",
            f"- {baseline['label']}: {baseline['system']['peak_rss_mib_after']:.1f} MiB",
            f"- {candidate['label']}: {candidate['system']['peak_rss_mib_after']:.1f} MiB",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = tuple(rows[0].keys()) if rows else ("stage",)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="MacBook benchmark.json")
    parser.add_argument("candidate", type=Path, help="Raspberry Pi benchmark.json")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/comparison"))
    args = parser.parse_args(argv)
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    warnings = compatibility_warnings(baseline, candidate)
    rows = comparison_rows(baseline, candidate)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    markdown = markdown_report(baseline, candidate, rows, warnings)
    (output / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    write_csv(rows, output / "comparison.csv")
    print(markdown)
    print(f"Comparison files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
