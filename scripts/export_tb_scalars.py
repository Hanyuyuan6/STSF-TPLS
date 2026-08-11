# -*- coding: utf-8 -*-
"""Export TensorBoard scalar events to one tidy CSV (run,tag,step,value).

The training-curve CSV was originally exported by hand and the
step was never scripted. This script IS that step, so the export is reproducible from the
.tfevents files it reads.

Usage (repo root):
    python -m scripts.export_tb_scalars \
        --logdir runs --runs rev_carvana_no_aux_s42 rev_carvana_fixed_s42 rev_carvana_tpls_s42 \
        --alias no_aux fixed tpls --out results/tb_scalars.csv

Each --runs entry is a subdirectory of --logdir containing .tfevents files
(the layout written by src/utils/tb_logger.py: log_dir/<experiment_name>/).
--alias gives the short run label written to the CSV's `run` column
(defaults to the run directory name). Every scalar tag is exported.
"""
import argparse
import csv
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_run(run_dir: Path, alias: str, writer: csv.writer) -> int:
    # EventAccumulator concatenates EVERY event file in the directory, and re-running a config
    # under the same experiment_name just adds another one. The exported CSV then carries one row
    # per event, i.e. repeated (tag, step) pairs -- curves that overlap exactly on a plot while the
    # row count is a multiple of the truth. Say so loudly instead of emitting it silently.
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(events) > 1:
        print(f"WARNING: {run_dir} holds {len(events)} .tfevents files (the same experiment_name "
              f"was run more than once); every (tag, step) will appear once per file -- deduplicate "
              f"before plotting, or export from a directory with a single event file",
              file=sys.stderr)
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    n = 0
    for tag in sorted(acc.Tags().get("scalars", [])):
        for ev in acc.Scalars(tag):
            writer.writerow([alias, tag, ev.step, ev.value])
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True, help="TensorBoard root (contains one subdir per run)")
    ap.add_argument("--runs", nargs="+", required=True, help="run subdirectory names under --logdir")
    ap.add_argument("--alias", nargs="*", default=None,
                    help="short labels for the CSV run column (same order as --runs)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    aliases = args.alias if args.alias else [r for r in args.runs]
    if len(aliases) != len(args.runs):
        ap.error("--alias must have the same length as --runs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "tag", "step", "value"])
        for run, alias in zip(args.runs, aliases):
            run_dir = Path(args.logdir) / run
            if not run_dir.is_dir():
                print(f"ERROR: no such run dir: {run_dir}", file=sys.stderr)
                return 1
            n = export_run(run_dir, alias, w)
            print(f"{alias:12s} <- {run_dir}  ({n} scalar points)")
            total += n
    print(f"wrote {out} ({total} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
