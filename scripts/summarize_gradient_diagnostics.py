"""Aggregate per-step TPLS gradient diagnostics without pooling repeat sizes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.gradient_diagnostics import (
    atomic_write_json,
    load_gradient_records,
    summarize_gradient_records,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="one JSONL file per instrumented repeat")
    parser.add_argument("--output", required=True, help="summary JSON path")
    parser.add_argument("--expected_repeats", type=int, default=None)
    args = parser.parse_args()

    runs = load_gradient_records(args.inputs)
    if args.expected_repeats is not None and len(runs) != args.expected_repeats:
        raise SystemExit(
            f"expected {args.expected_repeats} repeats, found {len(runs)}"
        )
    summary = summarize_gradient_records(runs)
    expected_phases = {"early", "middle", "late"}
    incomplete_runs = {
        label: sorted(set(summary["per_run_phase_means"][label]))
        for label in summary["per_run_phase_means"]
        if set(summary["per_run_phase_means"][label]) != expected_phases
    }
    if incomplete_runs:
        raise SystemExit(f"repeats with incomplete TPLS phases: {incomplete_runs}")
    actual_phases = set(summary["phases"])
    if actual_phases != expected_phases:
        raise SystemExit(
            f"expected phases {sorted(expected_phases)}, found {sorted(actual_phases)}"
        )
    atomic_write_json(args.output, summary)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
