"""Checkpoint loading with an explicit unsafe-pickle opt-in."""

from __future__ import annotations

import warnings

import torch


def load_checkpoint(path, *, map_location=None, allow_unsafe_pickle=False):
    """Load tensor/basic-type checkpoints, failing closed unless pickle is explicit.

    ``weights_only=True`` is a compatibility boundary, not a security sandbox for
    PyTorch releases affected by checkpoint-deserialization advisories. Checkpoints
    still need to come from a trusted source.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                f"Safe checkpoint loading failed for {path!s}. Refusing the unsafe pickle "
                "fallback. Only for a checkpoint you independently trust, rerun with "
                "--allow_unsafe_pickle."
            ) from exc
        warnings.warn(
            "Unsafe pickle checkpoint loading was explicitly enabled. A malicious checkpoint "
            "can execute arbitrary code.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)
