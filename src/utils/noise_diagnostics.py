"""Measurement-to-reconstruction noise amplification diagnostics."""

from __future__ import annotations

import math

import numpy as np


def _row_rms(array: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(array, dtype=np.float64), axis=1))


def _row_range(array: np.ndarray) -> np.ndarray:
    return np.max(array, axis=1) - np.min(array, axis=1)


def _minmax_rows(array: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return (array - np.min(array, axis=1, keepdims=True)) / ranges[:, None]


def _uint8_roundtrip(array: np.ndarray) -> np.ndarray:
    """Match the deployed grayscale PNG conversion and reload in [0, 1]."""
    return np.floor(np.clip(array, 0.0, 1.0) * 255.0 + 0.5) / 255.0


def noise_amplification_metrics(
    clean_measurements: np.ndarray,
    noisy_measurements: np.ndarray,
    hadamard_rows: np.ndarray,
    *,
    ta_uint8_roundtrip: bool = True,
) -> list[dict[str, float]]:
    """Measure relative perturbations before and after the fixed-physics lift.

    Reconstruction uses the scaled Hadamard adjoint ``s @ Phi / N`` followed by
    removal of the spatially uniform DC plane.  The returned ``linear_ratio`` is
    the direct realized-noise measurement; ``first_order_ratio`` is the first-order
    prediction ``sqrt(M) / N * R_IF / R_TA``.
    """
    clean = np.asarray(clean_measurements, dtype=np.float32)
    noisy = np.asarray(noisy_measurements, dtype=np.float32)
    phi = np.asarray(hadamard_rows, dtype=np.float32)
    if clean.ndim != 2 or noisy.shape != clean.shape:
        raise ValueError("clean and noisy measurements must have the same [B, M] shape")
    if phi.ndim != 2 or phi.shape[0] != clean.shape[1]:
        raise ValueError("hadamard_rows must have shape [M, N]")

    measurement_range = _row_range(clean)
    if np.any(measurement_range <= 0.0):
        raise ValueError("clean measurement range must be positive for every sample")

    n_pixels = phi.shape[1]
    clean_recon = (clean @ phi) / float(n_pixels)
    noisy_recon = (noisy @ phi) / float(n_pixels)
    clean_recon -= np.mean(clean_recon, axis=1, keepdims=True)
    noisy_recon -= np.mean(noisy_recon, axis=1, keepdims=True)
    reconstruction_range = _row_range(clean_recon)
    if np.any(reconstruction_range <= 0.0):
        raise ValueError("clean reconstruction range must be positive for every sample")

    measurement_noise_rms = _row_rms(noisy - clean)
    reconstruction_noise_rms = _row_rms(noisy_recon - clean_recon)
    relative_if = measurement_noise_rms / measurement_range
    relative_ta = reconstruction_noise_rms / reconstruction_range
    linear_ratio = relative_ta / relative_if
    first_order_ratio = (
        math.sqrt(clean.shape[1])
        / float(n_pixels)
        * measurement_range
        / reconstruction_range
    )

    clean_if_normalized = _minmax_rows(clean, measurement_range)
    noisy_measurement_range = _row_range(noisy)
    if np.any(noisy_measurement_range <= 0.0):
        raise ValueError("noisy measurement range must be positive for every sample")
    noisy_if_normalized = _minmax_rows(noisy, noisy_measurement_range)
    clean_ta_normalized = _minmax_rows(clean_recon, reconstruction_range)
    noisy_reconstruction_range = _row_range(noisy_recon)
    if np.any(noisy_reconstruction_range <= 0.0):
        raise ValueError("noisy reconstruction range must be positive for every sample")
    noisy_ta_normalized = _minmax_rows(noisy_recon, noisy_reconstruction_range)
    if ta_uint8_roundtrip:
        clean_ta_normalized = _uint8_roundtrip(clean_ta_normalized)
        noisy_ta_normalized = _uint8_roundtrip(noisy_ta_normalized)
    normalized_if_shift = _row_rms(noisy_if_normalized - clean_if_normalized)
    normalized_ta_shift = _row_rms(noisy_ta_normalized - clean_ta_normalized)

    records = []
    for index in range(clean.shape[0]):
        records.append(
            {
                "measurement_range": float(measurement_range[index]),
                "reconstruction_range": float(reconstruction_range[index]),
                "measurement_noise_rms": float(measurement_noise_rms[index]),
                "reconstruction_noise_rms": float(reconstruction_noise_rms[index]),
                "relative_if": float(relative_if[index]),
                "relative_ta": float(relative_ta[index]),
                "linear_ratio": float(linear_ratio[index]),
                "first_order_ratio": float(first_order_ratio[index]),
                "normalized_if_shift": float(normalized_if_shift[index]),
                "normalized_ta_shift": float(normalized_ta_shift[index]),
                "normalized_shift_ratio": float(
                    normalized_ta_shift[index] / normalized_if_shift[index]
                ),
            }
        )
    return records
