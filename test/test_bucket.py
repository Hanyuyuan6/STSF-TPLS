"""Bucket anchors — the forward physics + measurement-noise injection the released
regime-reversal protocol rests on.

Why this file exists: `src/utils/bucket.py` had **zero** test coverage (17 statements, 0
executed — not even imported by the suite) while being the mechanism behind the central
protocol.

HOW THESE ARE WRITTEN (read before editing): every test drives `compute_bucket_gpu` and
compares against an **independent** implementation written from the docstring's stated
formula — never a copy of the source, never a re-derivation inside the assert. The first
draft of this file did the latter and 3 of its 4 tests stayed GREEN under mutation of their
own target (SNR exponent /20->/10; `ac` no longer excluding DC; eps guard deleted). Each
test below is mutation-proven to go RED when its target breaks; keep it that way.

Run:  pytest test/ -v          (CPU-only; the GPU parity test skips without CUDA)
"""
import numpy as np
import pytest
import torch

from src.utils.bucket import build_phi, compute_bucket_gpu
from src.utils.ghost_patterns import get_hadamard_matrix

IMG, M = 32, 64          # small but real: N=1024 is a power of two, so Φ is exact
N = IMG * IMG
SEED = 1234


def _img(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 1, IMG, IMG, generator=g)


def _reference_bucket(image, phi, noise_snr_db=None, noise_ref='full', eps=1e-8, seed=None,
                      return_raw=False):
    """Independent transcription of bucket.py's DOCSTRING (not of its code):

        bucket_raw  = Φ · vec(img)
        sigma       = std(ref) · 10^(-SNR/20)      ref = raw[:,1:] if 'ac' else raw
        raw        += sigma · N(0,1)               # BEFORE normalisation
        bucket_norm = (raw - min) / (max - min + eps)     # per sample

    return_raw stops before the normalisation, so a caller can recover the noise that was
    actually drawn (noisy_raw - clean_raw). Without it the min-max hides the draw.
    """
    raw = image.reshape(image.size(0), -1).to(phi.dtype) @ phi.t()
    if noise_snr_db is not None:
        ref = raw[:, 1:] if noise_ref == 'ac' else raw
        sigma = ref.std(dim=1, keepdim=True) * (10.0 ** (-float(noise_snr_db) / 20.0))
        if seed is not None:
            torch.manual_seed(seed)
        raw = raw + sigma * torch.randn_like(raw)
    if return_raw:
        return raw
    mn = raw.amin(dim=1, keepdim=True)
    mx = raw.amax(dim=1, keepdim=True)
    return (raw - mn) / (mx - mn + eps)


def test_forward_matches_closed_form():
    """u-anchor: Φ from build_phi == the shipped Hadamard construction, and row 0 (DC) sums x."""
    phi = build_phi(IMG, M, 'cpu')
    x = _img()
    ref = torch.from_numpy(get_hadamard_matrix(N, M).astype(np.float32))
    assert torch.allclose(phi, ref, atol=0), 'build_phi must hand back the canonical Φ'
    raw = x.reshape(x.size(0), -1) @ phi.t()
    assert torch.allclose(raw[:, 0], x.reshape(x.size(0), -1).sum(1), atol=1e-3)  # DC row


def test_clean_bucket_matches_independent_reference():
    """Differential: the shipped clean path == the formula in its own docstring."""
    phi = build_phi(IMG, M, 'cpu')
    x = _img(batch=4, seed=1)
    assert torch.allclose(compute_bucket_gpu(x, phi), _reference_bucket(x, phi), atol=1e-6)


def test_normalisation_is_per_sample_minmax():
    """Each sample is min-max'd independently -> exactly 0 and 1 present per row."""
    out = compute_bucket_gpu(_img(batch=3), build_phi(IMG, M, 'cpu'))
    assert torch.allclose(out.amin(1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(out.amax(1), torch.ones(3), atol=1e-4)


@pytest.mark.parametrize('snr', [40, 30, 20, 10])
@pytest.mark.parametrize('ref_mode', ['ac', 'full'])
def test_noisy_bucket_matches_independent_reference(snr, ref_mode):
    """The dial the reversal turns. Pins BOTH the 10^(-SNR/20) amplitude convention and the
    'ac' -> exclude-row-0 semantics, by driving the real function under a fixed seed and
    comparing to the docstring's formula. An exponent slip (/20 -> /10) or an `ac` that stops
    dropping the DC row both change sigma -> this fails."""
    phi = build_phi(IMG, M, 'cpu')
    x = _img(batch=8, seed=2)
    torch.manual_seed(SEED)
    got = compute_bucket_gpu(x, phi, noise_snr_db=snr, noise_ref=ref_mode)
    exp = _reference_bucket(x, phi, noise_snr_db=snr, noise_ref=ref_mode, seed=SEED)
    assert torch.allclose(got, exp, atol=1e-6), f'max|Δ|={(got - exp).abs().max():.2e}'


def test_ac_and_full_references_actually_differ():
    """'ac' exists because natural-image buckets are DC-dominated: sigma('full') must exceed
    sigma('ac') materially. Same seed both sides, so any difference is the reference switch."""
    phi = build_phi(IMG, M, 'cpu')
    x = _img(batch=8, seed=5)
    torch.manual_seed(SEED)
    ac = compute_bucket_gpu(x, phi, noise_snr_db=20, noise_ref='ac')
    torch.manual_seed(SEED)
    full = compute_bucket_gpu(x, phi, noise_snr_db=20, noise_ref='full')
    assert not torch.allclose(ac, full, atol=1e-4), "'ac' and 'full' must not coincide"
    raw = x.reshape(x.size(0), -1) @ phi.t()
    assert float((raw.std(1) / raw[:, 1:].std(1)).mean()) > 1.05  # DC really does dominate


def test_realised_snr_equals_nominal():
    """Calibration anchor: the noise the docstring's formula ACTUALLY draws realises the
    nominal SNR — measured as (noisy_raw - clean_raw), never re-synthesised in the assert.

    An earlier version of this test recomputed `sigma * randn` inline and compared that to
    `nominal`; the signal std cancels algebraically, so it held for any implementation and
    was worth nothing. Subtracting two raws is what makes it bite: mutate the exponent in
    _reference_bucket (/20 -> /10) and the realised value doubles.
    """
    phi = build_phi(IMG, M, 'cpu')
    x = _img(batch=64, seed=3)
    clean_raw = _reference_bucket(x, phi, return_raw=True)
    for nominal in (40, 30, 20, 10):
        noisy_raw = _reference_bucket(x, phi, noise_snr_db=nominal, noise_ref='ac',
                                      seed=SEED, return_raw=True)
        drawn = noisy_raw - clean_raw                       # the noise actually injected
        realised = float((20 * torch.log10(clean_raw[:, 1:].std(1) / drawn.std(1))).mean())
        assert abs(realised - nominal) < 1.0, f'nominal {nominal} dB -> realised {realised:.2f}'


def test_noise_is_injected_before_normalisation():
    """Ordering IS the mechanism claim: noise enters the raw bucket, the min-max then
    renormalises around it. If it were applied after, the output would no longer be an exact
    per-sample [0,1] min-max."""
    phi = build_phi(IMG, M, 'cpu')
    x = _img(batch=8, seed=7)
    torch.manual_seed(SEED)
    noisy = compute_bucket_gpu(x, phi, noise_snr_db=0, noise_ref='ac')
    assert torch.allclose(noisy.amin(1), torch.zeros(8), atol=1e-6)
    assert torch.allclose(noisy.amax(1), torch.ones(8), atol=1e-4)
    assert (noisy - compute_bucket_gpu(x, phi)).abs().max() > 0.05


def test_degenerate_input_hits_the_eps_guard():
    """An all-zero image drives raw to all-zeros -> mx == mn -> 0/0 without the eps guard."""
    phi = build_phi(IMG, M, 'cpu')
    raw = torch.zeros(1, N) @ phi.t()
    assert float(raw.amax() - raw.amin()) == 0.0, 'premise: this input really is degenerate'
    out = compute_bucket_gpu(torch.zeros(1, 1, IMG, IMG), phi)
    assert torch.isfinite(out).all(), 'eps guard must keep a degenerate sample finite'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cpu_gpu_parity_at_the_documented_tolerance():
    """bucket.py's module docstring states the CPU twin (base_dataset.py) is followed
    'verbatim identical ... allclose<1e-4'. Nothing checked it. If the two paths drift,
    every reported number moves silently."""
    x = _img(batch=4, seed=11)
    cpu = compute_bucket_gpu(x, build_phi(IMG, M, 'cpu'))
    gpu = compute_bucket_gpu(x.cuda(), build_phi(IMG, M, 'cuda')).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4), f'max|Δ|={(cpu - gpu).abs().max():.2e}'
