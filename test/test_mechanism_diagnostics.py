import math
import json

import numpy as np
import pytest
import torch
from scipy.linalg import hadamard

from src.utils.gradient_diagnostics import (
    gradient_metrics,
    measure_loss_gradients,
    summarize_gradient_records,
)
from src.utils.noise_diagnostics import noise_amplification_metrics
from src.utils.ghost_patterns import get_hadamard_matrix
from src.trainer import SegmentationTrainer


def test_gradient_metrics_include_missing_parameters_as_zero():
    metrics = gradient_metrics(
        [torch.tensor([3.0, 4.0]), None],
        [torch.tensor([4.0, -3.0]), torch.tensor([0.0])],
        seg_weight=0.1,
        aux_weight=0.9,
    )
    assert metrics["cosine"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["seg_grad_norm"] == pytest.approx(5.0)
    assert metrics["aux_grad_norm"] == pytest.approx(5.0)
    assert metrics["raw_magnitude_ratio"] == pytest.approx(1.0)
    assert metrics["effective_pull"] == pytest.approx(9.0)
    assert metrics["magnitude_similarity"] == pytest.approx(1.0)


def test_measuring_gradients_does_not_populate_parameter_grad():
    weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    seg_loss = (weight[0] + weight[1]) ** 2
    aux_loss = (weight[0] - weight[1]) ** 2
    metrics = measure_loss_gradients(
        seg_loss,
        aux_loss,
        [weight],
        seg_weight=0.5,
        aux_weight=0.5,
    )
    assert weight.grad is None
    assert metrics["seg_grad_norm"] > 0.0
    (0.5 * seg_loss + 0.5 * aux_loss).backward()
    assert weight.grad is not None


def test_gradient_summary_equal_weights_repeats_not_steps():
    base = {
        "phase": "early",
        "cosine": 0.0,
        "effective_pull": 1.0,
        "magnitude_similarity": 1.0,
    }
    runs = {
        "a": [
            {**base, "run_label": "a", "raw_magnitude_ratio": 1.0},
            {**base, "run_label": "a", "raw_magnitude_ratio": 3.0},
        ],
        "b": [{**base, "run_label": "b", "raw_magnitude_ratio": 10.0}],
    }
    summary = summarize_gradient_records(runs)
    # Per-run means are 2 and 10; equal weighting gives 6 (pooled steps would give 14/3).
    assert summary["phases"]["early"]["raw_magnitude_ratio"]["mean_over_runs"] == 6.0
    assert summary["phases"]["early"]["negative_cosine_probability"]["mean_over_runs"] == 0.0


def test_gradient_summary_reports_probability_of_conflicting_steps():
    base = {
        "phase": "middle",
        "effective_pull": 1.0,
        "magnitude_similarity": 1.0,
        "raw_magnitude_ratio": 1.0,
    }
    runs = {
        "a": [
            {**base, "run_label": "a", "cosine": -0.2},
            {**base, "run_label": "a", "cosine": 0.4},
        ],
        "b": [
            {**base, "run_label": "b", "cosine": -0.1},
            {**base, "run_label": "b", "cosine": -0.3},
        ],
    }
    summary = summarize_gradient_records(runs)
    assert summary["per_run_phase_means"]["a"]["middle"]["negative_cosine_probability"] == 0.5
    assert summary["per_run_phase_means"]["b"]["middle"]["negative_cosine_probability"] == 1.0
    assert summary["phases"]["middle"]["negative_cosine_probability"]["mean_over_runs"] == 0.75


def test_gradient_summary_rejects_any_missing_or_nonfinite_step_metric():
    record = {
        "run_label": "a", "phase": "early", "cosine": 0.1,
        "raw_magnitude_ratio": None, "effective_pull": 1.0,
        "magnitude_similarity": 1.0,
    }
    with pytest.raises(ValueError, match="must be finite for every optimizer step"):
        summarize_gradient_records({"a": [record]})


def test_supplement_stepwise_phase_counts_are_exact():
    assert SegmentationTrainer._stepwise_boundaries(9 * 40, 0.3, 0.3) == (108, 216)


def test_noise_amplification_matches_equation7_for_orthogonal_rows():
    phi = hadamard(4).astype(np.float32)
    clean = np.array([[10.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    measurement_noise = np.array([[0.0, 1.0, -1.0, 1.0]], dtype=np.float32)
    record = noise_amplification_metrics(clean, clean + measurement_noise, phi)[0]
    assert record["relative_if"] > 0.0
    assert record["relative_ta"] > 0.0
    assert math.isfinite(record["normalized_shift_ratio"])
    assert record["linear_ratio"] == pytest.approx(record["equation7_ratio"], rel=1e-6)


def test_noise_amplification_ta_shift_uses_uint8_roundtrip():
    phi = hadamard(4).astype(np.float32)
    clean = np.array([[10.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    noisy = clean + np.array([[0.0, 0.01, -0.01, 0.01]], dtype=np.float32)
    quantized = noise_amplification_metrics(clean, noisy, phi)[0]
    floating = noise_amplification_metrics(
        clean, noisy, phi, ta_uint8_roundtrip=False
    )[0]
    assert quantized["normalized_ta_shift"] != floating["normalized_ta_shift"]


def test_noise_amplification_rejects_shape_mismatch():
    with pytest.raises(ValueError, match=r"same \[B, M\] shape"):
        noise_amplification_metrics(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            np.eye(4, dtype=np.float32),
        )


@pytest.mark.parametrize("n,m", [(12, 4), (16, 0), (16, 17)])
def test_hadamard_rejects_nonorthogonal_or_out_of_range_shapes(n, m):
    with pytest.raises(ValueError):
        get_hadamard_matrix(n, m)


def test_trainer_diagnostics_preserve_optimizer_step_and_limit(tmp_path):
    class TinyDualHead(torch.nn.Module):
        input_type = "image"

        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Conv2d(1, 2, 1)
            self.seg = torch.nn.Conv2d(2, 1, 1)
            self.aux = torch.nn.Conv2d(2, 1, 1)

        def forward(self, value):
            shared = torch.tanh(self.shared(value))
            return {"logits": self.seg(shared), "aux_recon": self.aux(shared)}

    model = TinyDualHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "image": torch.ones(2, 1, 2, 2),
        "mask": torch.ones(2, 2, 2, dtype=torch.long),
    }
    diagnostic_path = tmp_path / "gradient.jsonl"
    config = {
        "data": {"classes": 1, "bucket_on_gpu": False, "bucket_noise_snr_db": None},
        "training": {
            "epochs": 9,
            "amp": False,
            "gradient_clip": 0.0,
            "max_steps_per_epoch": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "experiment_name": "tiny",
            "loss": {
                "seg_weight": 1.0,
                "aux_recon_weight": 1.0,
                "adaptive_loss": {
                    "enable": True,
                    "stage1_ratio": 0.3,
                    "stage2_ratio": 0.3,
                    "stage1_weights": [0.1, 0.9],
                    "stage2_weights": [0.5, 0.5],
                    "stage3_weights": [0.9, 0.1],
                },
            },
            "gradient_diagnostics": {
                "jsonl": str(diagnostic_path),
                "run_label": "tiny",
                "seed": 42,
            },
        },
        "logging": {"use_tensorboard": False},
        "inference": {"threshold": 0.5},
        "model": {"name": "TinyDualHead"},
    }
    seg_loss = lambda logits, mask: torch.mean((logits.squeeze(1) - mask.float()) ** 2)
    aux_loss = lambda recon, image: torch.mean((recon - image) ** 2)
    before = model.shared.weight.detach().clone()
    trainer = SegmentationTrainer(
        model,
        optimizer,
        seg_loss,
        aux_loss,
        [batch, batch],
        [batch],
        None,
        torch.device("cpu"),
        config,
    )
    trainer._train_epoch(1, 0.1, 0.9)
    records = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["phase"] == "early"
    assert records[0]["effective_pull"] is not None
    assert not torch.equal(before, model.shared.weight.detach())
