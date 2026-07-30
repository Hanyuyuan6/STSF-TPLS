"""Utilities for measuring and summarizing TPLS gradient dynamics."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch


def gradient_metrics(
    seg_grads: Sequence[torch.Tensor | None],
    aux_grads: Sequence[torch.Tensor | None],
    *,
    seg_weight: float,
    aux_weight: float,
) -> dict[str, float | None]:
    """Compute unweighted gradient geometry and the runtime-weighted pull ratio.

    ``None`` gradients are treated as zero for the full parameter vector.  This is
    the mathematical gradient of a loss with respect to parameters that the loss
    does not use (for example, the task-specific output heads).
    """
    if len(seg_grads) != len(aux_grads):
        raise ValueError("seg_grads and aux_grads must have the same length")

    dot_tensor = None
    seg_sq_tensor = None
    aux_sq_tensor = None
    for seg_grad, aux_grad in zip(seg_grads, aux_grads):
        if seg_grad is not None:
            seg = seg_grad.detach().float()
            term = torch.sum(seg * seg)
            seg_sq_tensor = term if seg_sq_tensor is None else seg_sq_tensor + term
        if aux_grad is not None:
            aux = aux_grad.detach().float()
            term = torch.sum(aux * aux)
            aux_sq_tensor = term if aux_sq_tensor is None else aux_sq_tensor + term
        if seg_grad is not None and aux_grad is not None:
            term = torch.sum(seg_grad.detach().float() * aux_grad.detach().float())
            dot_tensor = term if dot_tensor is None else dot_tensor + term

    seg_sq = float(seg_sq_tensor.item()) if seg_sq_tensor is not None else 0.0
    aux_sq = float(aux_sq_tensor.item()) if aux_sq_tensor is not None else 0.0
    dot = float(dot_tensor.item()) if dot_tensor is not None else 0.0

    seg_norm = math.sqrt(seg_sq)
    aux_norm = math.sqrt(aux_sq)
    denom = seg_norm * aux_norm
    cosine = dot / denom if denom > 0.0 else None
    raw_ratio = aux_norm / seg_norm if seg_norm > 0.0 else None
    weighted_denom = float(seg_weight) * seg_norm
    effective_pull = (
        float(aux_weight) * aux_norm / weighted_denom
        if weighted_denom > 0.0
        else None
    )
    magnitude_similarity = (
        2.0 * seg_norm * aux_norm / (seg_sq + aux_sq)
        if seg_sq + aux_sq > 0.0
        else None
    )
    return {
        "cosine": cosine,
        "seg_grad_norm": seg_norm,
        "aux_grad_norm": aux_norm,
        "raw_magnitude_ratio": raw_ratio,
        "effective_pull": effective_pull,
        "magnitude_similarity": magnitude_similarity,
    }


def measure_loss_gradients(
    seg_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    seg_weight: float,
    aux_weight: float,
) -> dict[str, float | None]:
    """Measure each loss separately without changing the subsequent optimizer step."""
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    seg_grads = torch.autograd.grad(
        seg_loss, params, retain_graph=True, allow_unused=True
    )
    aux_grads = torch.autograd.grad(
        aux_loss, params, retain_graph=True, allow_unused=True
    )
    return gradient_metrics(
        seg_grads,
        aux_grads,
        seg_weight=seg_weight,
        aux_weight=aux_weight,
    )


def append_jsonl(path: str | Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, allow_nan=False)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")


def load_gradient_records(paths: Iterable[str | Path]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    required = {
        "run_label",
        "phase",
        "cosine",
        "raw_magnitude_ratio",
        "effective_pull",
    }
    for raw_path in paths:
        path = Path(raw_path)
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = required.difference(record)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields: {sorted(missing)}"
                    )
                records.append(record)
        if not records:
            raise ValueError(f"gradient diagnostic file is empty: {path}")
        labels = {str(record["run_label"]) for record in records}
        if len(labels) != 1:
            raise ValueError(f"{path} contains multiple run labels: {sorted(labels)}")
        label = labels.pop()
        if label in runs:
            raise ValueError(f"duplicate run_label across inputs: {label}")
        runs[label] = records
    return runs


def summarize_gradient_records(runs: dict[str, list[dict]]) -> dict:
    """Average within each run first, then summarize equally weighted repeats."""
    metrics = ("cosine", "raw_magnitude_ratio", "effective_pull", "magnitude_similarity")
    per_run: dict[str, dict[str, dict[str, float]]] = {}
    phase_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for label, records in sorted(runs.items()):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            grouped[str(record["phase"])].append(record)
        per_run[label] = {}
        for phase, phase_records in sorted(grouped.items()):
            per_run[label][phase] = {}
            for metric in metrics:
                values = []
                for step_index, record in enumerate(phase_records, start=1):
                    value = record.get(metric)
                    if value is None or not math.isfinite(float(value)):
                        raise ValueError(
                            f"run {label}, phase {phase}, step {step_index}: "
                            f"{metric} must be finite for every optimizer step"
                        )
                    values.append(float(value))
                mean = sum(values) / len(values)
                per_run[label][phase][metric] = mean
                phase_values[phase][metric].append(mean)

            cosine_values = [float(record["cosine"]) for record in phase_records]
            negative_probability = sum(value < 0.0 for value in cosine_values) / len(
                cosine_values
            )
            per_run[label][phase]["negative_cosine_probability"] = negative_probability
            phase_values[phase]["negative_cosine_probability"].append(
                negative_probability
            )
            per_run[label][phase]["step_count"] = len(phase_records)

    phases: dict[str, dict] = {}
    for phase, metric_values in sorted(phase_values.items()):
        phases[phase] = {}
        for metric, values in sorted(metric_values.items()):
            phases[phase][metric] = {
                "mean_over_runs": sum(values) / len(values),
                "min_run_mean": min(values),
                "max_run_mean": max(values),
                "run_count": len(values),
            }
    return {
        "aggregation": "mean within run and phase, then equal-weight mean across runs",
        "run_count": len(runs),
        "phases": phases,
        "per_run_phase_means": per_run,
    }


def atomic_write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
