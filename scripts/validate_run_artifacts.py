"""Strictly validate run_all outputs and publish a SHA-256 manifest."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

SEGMENTATION_METRICS = {'pa', 'mpa', 'miou', 'mdice', 'miou_fg', 'mdice_fg'}

def validate_artifact(spec, root=Path('.')):
    """Validate ``path|split|dataset|seed|kind|model|experiment`` metadata."""
    try:
        (
            relpath, expected_split, expected_dataset, expected_seed, kind,
            expected_model, expected_experiment,
        ) = spec.split('|')
    except ValueError as exc:
        raise ValueError(f'Invalid artifact spec {spec!r}') from exc
    path = root / relpath
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot parse expected artifact {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'{path}: top-level JSON must be an object')

    expected = {
        'split': expected_split,
        'dataset': expected_dataset,
        'train_seed': int(expected_seed),
        'model': expected_model,
        'experiment_name': expected_experiment,
    }
    for field, wanted in expected.items():
        if data.get(field) != wanted:
            raise ValueError(
                f'{path}: {field}={data.get(field)!r}, expected {wanted!r}')
    samples = data.get('samples')
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError(f'{path}: samples must be a positive integer, got {samples!r}')
    metrics = data.get('metrics')
    if not isinstance(metrics, dict) or set(metrics) != SEGMENTATION_METRICS:
        raise ValueError(
            f'{path}: metrics must contain exactly {sorted(SEGMENTATION_METRICS)}')
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f'{path}: metric {name!r} is not a finite number')
    if kind.startswith('recon:'):
        wanted_method = kind.split(':', 1)[1]
        if data.get('method') != wanted_method:
            raise ValueError(
                f'{path}: method={data.get("method")!r}, expected {wanted_method!r}')
    elif kind != 'segmentation':
        raise ValueError(f'{path}: unknown artifact kind {kind!r}')

    return {
        'path': relpath.replace('\\', '/'),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'split': expected_split,
        'dataset': expected_dataset,
        'train_seed': int(expected_seed),
        'samples': samples,
        'kind': kind,
        'model': expected_model,
        'experiment_name': expected_experiment,
    }


def write_manifest(mode, specs, manifest_path, root=Path('.'), expected_count=None):
    if len(specs) != len(set(specs)):
        raise ValueError('Expected artifact list contains duplicates')
    if expected_count is not None and len(specs) != expected_count:
        raise ValueError(
            f'Expected {expected_count} artifact specs, received {len(specs)}')
    artifacts = [validate_artifact(spec, root=root) for spec in specs]
    output_root = (root / manifest_path).parent
    expected_paths = {(root / spec.split('|', 1)[0]).resolve() for spec in specs}
    actual_paths = {
        path.resolve() for path in output_root.rglob('*.json')
        if path.resolve() != (root / manifest_path).resolve()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f'Artifact JSON inventory mismatch; missing={missing}, extra={extra}')
    payload = {
        'schema_version': 2,
        'mode': mode,
        'expected_artifact_count': len(specs),
        'artifacts': artifacts,
    }
    path = root / manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['smoke', 'full'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--expected-count', required=True, type=int)
    parser.add_argument('--artifact', action='append', required=True,
                        help='path|split|dataset|seed|kind|model|experiment')
    args = parser.parse_args()
    payload = write_manifest(
        args.mode, args.artifact, Path(args.manifest), expected_count=args.expected_count)
    print(f"validated {payload['expected_artifact_count']} artifacts -> {args.manifest}")


if __name__ == '__main__':
    main()
