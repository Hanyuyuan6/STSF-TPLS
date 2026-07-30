"""Run the five-repeat, shortened MNIST TPLS gradient protocol from Supplement S6."""

import argparse
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.gradient_diagnostics import (
    atomic_write_json,
    load_gradient_records,
    summarize_gradient_records,
)


def _git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
        untracked = sorted(
            os.fsdecode(value) for value in untracked_raw.split(b"\0") if value
        )
        snapshot = hashlib.sha256()
        snapshot.update(b"tracked-diff\0")
        snapshot.update(diff)
        for relative in untracked:
            path = Path(relative)
            content = path.read_bytes()
            snapshot.update(b"untracked\0")
            snapshot.update(os.fsencode(relative))
            snapshot.update(b"\0")
            snapshot.update(hashlib.sha256(content).digest())
        return {
            "commit": commit,
            "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "working_tree_snapshot_sha256": snapshot.hexdigest(),
            "untracked_files": untracked,
            "working_tree_dirty": bool(diff or untracked),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "working_tree_diff_sha256": None,
            "working_tree_snapshot_sha256": None,
            "untracked_files": None,
            "working_tree_dirty": None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/rev_mnist_tpls.yaml")
    parser.add_argument("--output_dir", default="results/gradient_diagnostics")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--steps_per_epoch", type=int, default=40)
    args = parser.parse_args()

    if args.repeats <= 0 or args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise SystemExit("repeats, epochs, and steps_per_epoch must all be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_paths = [output_dir / f"repeat_{index:02d}.jsonl" for index in range(1, args.repeats + 1)]
    reserved = [*jsonl_paths, output_dir / "summary.json", output_dir / "protocol.json"]
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise SystemExit(
            "refusing to mix with an existing diagnostic run; choose a new --output_dir: "
            + ", ".join(existing)
        )

    commands = []
    for index, jsonl_path in enumerate(jsonl_paths, start=1):
        label = f"gradient_r{index:02d}"
        command = [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--max_steps_per_epoch",
            str(args.steps_per_epoch),
            "--refuse_existing_output",
            "--gradient_diagnostics_jsonl",
            str(jsonl_path),
            "--gradient_stepwise_schedule",
            "--run_label",
            label,
        ]
        commands.append(command)
        subprocess.run(command, check=True)
        record_count = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())
        expected = args.epochs * args.steps_per_epoch
        if record_count != expected:
            raise RuntimeError(
                f"{jsonl_path}: expected {expected} optimizer-step records, found {record_count}"
            )

    runs = load_gradient_records(jsonl_paths)
    total_steps = args.epochs * args.steps_per_epoch
    stage1_end = round(total_steps * 0.3)
    stage2_end = round(total_steps * 0.6)
    expected_phase_counts = {
        "early": stage1_end,
        "middle": stage2_end - stage1_end,
        "late": total_steps - stage2_end,
    }
    observed_phase_counts = {
        label: dict(Counter(str(record["phase"]) for record in records))
        for label, records in runs.items()
    }
    wrong_counts = {
        label: counts
        for label, counts in observed_phase_counts.items()
        if counts != expected_phase_counts
    }
    if wrong_counts:
        raise RuntimeError(
            f"unexpected per-phase optimizer-step counts; expected "
            f"{expected_phase_counts}, found {wrong_counts}"
        )
    summary = summarize_gradient_records(runs)
    incomplete_runs = {
        label: sorted(set(summary["per_run_phase_means"][label]))
        for label in summary["per_run_phase_means"]
        if set(summary["per_run_phase_means"][label]) != {"early", "middle", "late"}
    }
    if incomplete_runs:
        raise RuntimeError(f"repeats with incomplete TPLS phases: {incomplete_runs}")
    if set(summary["phases"]) != {"early", "middle", "late"}:
        raise RuntimeError(f"incomplete TPLS phases: {sorted(summary['phases'])}")
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(
        output_dir / "protocol.json",
        {
            "protocol": "Supplement S6 shortened MNIST TPLS gradient diagnostic",
            "config": args.config,
            "seed": args.seed,
            "repeats": args.repeats,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "expected_phase_step_counts": expected_phase_counts,
            "python": sys.version,
            "torch": torch.__version__,
            "git": _git_state(),
            "commands": commands,
            "aggregation": summary["aggregation"],
        },
    )
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
