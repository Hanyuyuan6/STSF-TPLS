"""Anchor tests — known-answer and differential checks on the correctness primitives
the reported metrics rest on, run in CI so a silent regression cannot pass unnoticed.

These pin the *correctness primitives* the reported metrics rest on:
  - foreground mIoU/mDice aggregation vs an independent confusion matrix (differential),
  - the exact known answers (perfect prediction → 1, WBC gray-map, Hadamard orthogonality),
  - the per-architecture parameter counts claimed in the paper (contract layer).
Run:  pytest test/ -v   (needs the full-deps env; src.models pulls torch/kornia/tensorboard).
"""
import os

import numpy as np
import pytest

from src.datasets.specific_datasets import canonical_gray_map
from src.metrics.segmentation_metrics import batch_segmentation_metrics
from src.utils.ghost_patterns import get_hadamard_matrix


def test_fg_metric_matches_independent_reference():
    """Shipped fg mIoU/mDice must equal an independent numpy confusion-matrix
    computation on random inputs — guards against wrong class-averaging / bg inclusion."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        C = int(rng.integers(2, 4))
        pred = rng.integers(0, C, size=(3, 32, 32))
        gt = rng.integers(0, C, size=(3, 32, 32))
        got = batch_segmentation_metrics(pred, gt, num_classes=C)
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(gt.ravel(), pred.ravel()):
            cm[t, p] += 1
        iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-8)
        dice = 2 * np.diag(cm) / (cm.sum(1) + cm.sum(0) + 1e-8)
        assert abs(got["miou_fg"] - float(np.nanmean(iou[1:]))) < 1e-6
        assert abs(got["mdice_fg"] - float(np.nanmean(dice[1:]))) < 1e-6


def test_identical_prediction_is_perfect():
    """A perfect prediction must score fg mIoU = fg mDice = 1 (metric zero/one point)."""
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 3, size=(2, 24, 24))
    m = batch_segmentation_metrics(gt.copy(), gt.copy(), num_classes=3)
    assert abs(m["miou_fg"] - 1.0) < 1e-6
    assert abs(m["mdice_fg"] - 1.0) < 1e-6


def test_wbc_canonical_gray_map():
    """WBC mask labels come from nearest-canonical gray levels. Clean 0/128/255 → 0/1/2;
    stray values below 128 snap to background, never displacing the real cytoplasm(128)/nucleus(255).

    Imports the SHIPPED mapper — do not re-implement it here. A prior version defined its own
    `wmap` in this body and therefore stayed green when specific_datasets.py was mutated back to
    the known rank-of-unique bug (mutation-proven 2026-07-17). The assertions were always right;
    only the wiring was missing.
    """
    assert canonical_gray_map([0, 128, 255], 3).tolist() == [0, 1, 2]
    d = canonical_gray_map([0, 2, 7, 25, 128, 255], 3)   # Dataset2/034-style stray noise
    assert d[4] == 1 and d[5] == 2 and d[1:4].tolist() == [0, 0, 0]
    # the bug this guards: rank-of-unique would map the 6 distinct values to 0..5
    assert d.tolist() != [0, 1, 2, 3, 4, 5]


def test_hadamard_rows_orthogonal_exact():
    """The first-M natural-order Hadamard rows are ±1 and orthogonal, so the
    Gram matrix is EXACTLY N_cols * I (integer, zero eps) — the sensing operator is well-posed."""
    N, M = 128 * 128, 512
    H = get_hadamard_matrix(N, M).astype(np.float64)
    assert np.abs(H @ H.T - N * np.eye(M)).max() == 0.0


@pytest.mark.parametrize("cfg,claim,tol", [
    ("rev_carvana_tpls", 36.8, 0.15),          # STSF (GRUUNetPP) — 36.8481M, paper Sec. IV "Compared methods"
    ("rev_carvana_fcn", 30.2, 0.15),           # SPIFS (FCNUNetPP)
    ("rev_carvana_traditional", 29.4, 0.15),   # U-Net++ (BaselineUNetPP)
])
def test_param_counts_match_paper(cfg, claim, tol):
    """Each architecture instantiated from its shipped config must have the
    parameter count claimed in the paper (contract layer, pins Sec. IV capacity)."""
    import yaml
    import src.models as models
    p = os.path.join(os.path.dirname(__file__), "..", "configs", "experiments", cfg + ".yaml")
    c = yaml.safe_load(open(p, encoding="utf-8"))
    model = getattr(models, c["model"]["name"])(**c["model"]["params"])
    n = sum(pp.numel() for pp in model.parameters()) / 1e6
    assert abs(n - claim) <= tol, f"{cfg}={n:.2f}M expected {claim}±{tol}"
