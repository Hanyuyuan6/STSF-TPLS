"""Fail-closed inventory validation for reconstruction-backed datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reconstruction_inventory(split_dir: str | Path) -> list[dict[str, str]]:
    """Hash an exact, flat ``images/*.png`` + ``masks/*.png`` pair inventory."""
    split = Path(split_dir)
    by_kind: dict[str, dict[str, Path]] = {}
    for kind in ("images", "masks"):
        directory = split / kind
        if not directory.is_dir():
            raise RuntimeError(f"reconstruction directory is missing: {directory}")
        entries = sorted(directory.iterdir(), key=lambda value: value.name)
        invalid = [
            entry.name
            for entry in entries
            if entry.is_symlink() or not entry.is_file() or entry.suffix.lower() != ".png"
        ]
        if invalid:
            raise RuntimeError(
                f"unexpected or non-regular reconstruction files in {directory}: {invalid[:5]}"
            )
        stems = {entry.stem: entry for entry in entries}
        if len(stems) != len(entries):
            raise RuntimeError(f"duplicate reconstruction stems in {directory}")
        by_kind[kind] = stems

    image_stems = set(by_kind["images"])
    mask_stems = set(by_kind["masks"])
    if not image_stems or image_stems != mask_stems:
        raise RuntimeError(
            "reconstruction image/mask inventory is empty or has unmatched stems"
        )

    return [
        {
            "stem": stem,
            "image": f"images/{by_kind['images'][stem].name}",
            "image_sha256": sha256_file(by_kind["images"][stem]),
            "mask": f"masks/{by_kind['masks'][stem].name}",
            "mask_sha256": sha256_file(by_kind["masks"][stem]),
        }
        for stem in sorted(image_stems)
    ]


def inventory_sha256(inventory: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reconstruction_manifest(
    split_dir: str | Path, *, expected_signature: str | None = None
) -> dict:
    """Require a complete manifest whose hashes exactly match every PNG pair."""
    split = Path(split_dir)
    meta_path = split / "_dump_meta.json"
    try:
        manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read reconstruction manifest {meta_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"reconstruction manifest must be an object: {meta_path}")
    if manifest.get("complete") is not True or manifest.get("status") != "complete":
        raise RuntimeError(f"reconstruction manifest is not complete: {meta_path}")
    if expected_signature is not None and manifest.get("generation_signature") != expected_signature:
        raise RuntimeError(f"reconstruction generation signature mismatch: {meta_path}")

    total = manifest.get("total_in_split")
    processed = manifest.get("processed")
    if (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or processed != total
    ):
        raise RuntimeError(f"reconstruction processed/total contract is invalid: {meta_path}")

    actual = build_reconstruction_inventory(split)
    if len(actual) != total or manifest.get("pair_count") != total:
        raise RuntimeError(f"reconstruction pair count does not match the split total: {meta_path}")
    if manifest.get("file_count") != 2 * total:
        raise RuntimeError(f"reconstruction file count is invalid: {meta_path}")
    if manifest.get("inventory") != actual:
        raise RuntimeError(f"reconstruction file inventory or SHA-256 mismatch: {meta_path}")
    if manifest.get("inventory_sha256") != inventory_sha256(actual):
        raise RuntimeError(f"reconstruction inventory digest mismatch: {meta_path}")
    return manifest
