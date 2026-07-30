# -*- coding: utf-8 -*-
"""Merge clean and noise evaluation JSONs into the canonical results CSV.

The input CSV is never overwritten. By default this scans ``incoming/``,
``checkpoints/*/eval/``, ``results/``, and ``_rev/results/`` and writes
``results/master_results_merged.csv``. Rows are deduplicated by experiment id.
"""

import csv
import glob
import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [
    'experiment', 'dataset', 'model', 'family', 'M', 'rate_pct',
    'train_noise_db', 'eval_noise_db', 'perm_seed', 'seed', 'env',
    'miou_fg', 'mdice_fg', 'miou', 'mdice', 'pa', 'mpa', 'source',
    'ckpt_path',
]


def pct(metrics, key):
    value = metrics.get(key)
    return '' if value is None else f'{value * 100:.4f}'


def _base_row(experiment, dataset, model, family, metrics, source, ckpt,
              bucket_size=512, seed='', eval_noise_db='', train_noise_db='',
              perm_seed='', environment='review2026-07'):
    rate = 100.0 * int(bucket_size) / (128 * 128)
    return {
        'experiment': experiment,
        'dataset': dataset,
        'model': model,
        'family': family,
        'M': str(bucket_size),
        'rate_pct': f'{rate:.4f}'.rstrip('0').rstrip('.'),
        'train_noise_db': str(train_noise_db),
        'eval_noise_db': str(eval_noise_db),
        'perm_seed': str(perm_seed) if perm_seed is not None else '',
        'seed': str(seed),
        'env': environment,
        'miou_fg': pct(metrics, 'miou_fg'),
        'mdice_fg': pct(metrics, 'mdice_fg'),
        'miou': pct(metrics, 'miou'),
        'mdice': pct(metrics, 'mdice'),
        'pa': pct(metrics, 'pa'),
        'mpa': pct(metrics, 'mpa'),
        'source': source,
        'ckpt_path': ckpt,
    }


def clean_eval_row(data, source):
    """Map a released clean-evaluation JSON to one CSV row, or return None."""
    metrics = data.get('metrics')
    if not isinstance(metrics, dict) or data.get('noise_snr_db') is not None:
        return None

    source_stem = os.path.splitext(os.path.basename(source))[0]
    source_stem = re.sub(r'_test$', '', source_stem)
    declared = data.get('experiment') or data.get('experiment_name') or ''
    experiment = source_stem if re.search(r'_s\d+', source_stem) else declared
    experiment = experiment or source_stem

    patterns = [
        (r'^rev_(\w+?)_(traditional|fcn|no_aux|fixed|tpls)(?:_m\d+)?(?:_s\d+)?$', 'main_clean'),
        (r'^lift_(\w+?)_(\w+?)(?:_s\d+)?$', 'lift_ablation_clean'),
        (r'^recon_(\w+?)_(tradgi|admm-l1)(?:_s\d+)?$', 'reconstruction_clean'),
        (r'^ta_(\w+?)_(hsi|cs)(?:_\w+)?(?:_s\d+)?$', 'task_adapted_clean'),
    ]
    match = family = None
    for regex, candidate_family in patterns:
        match = re.match(regex, experiment)
        if match:
            family = candidate_family
            break
    if match is None:
        return None

    dataset, model_token = match.group(1), match.group(2)
    if family == 'lift_ablation_clean':
        model = f'lift_{model_token}'
    elif family == 'task_adapted_clean':
        model = f'ta_{model_token}'
    else:
        model = model_token

    seed_match = re.search(r'_s(\d+)', experiment)
    seed = data.get('train_seed', seed_match.group(1) if seed_match else '')
    m_match = re.search(r'_m(\d+)', experiment)
    bucket_size = data.get('bucket_size') or (int(m_match.group(1)) if m_match else 512)
    return _base_row(
        experiment, data.get('dataset') or dataset, model, family, metrics,
        os.path.basename(source), data.get('ckpt', ''), bucket_size=bucket_size,
        seed=seed, perm_seed=data.get('perm_seed'),
    )


def noise_eval_row(data, source):
    """Map a supported AC-referenced noise JSON to one row, or return None."""
    metrics = data.get('metrics')
    snr = data.get('noise_snr_db')
    if not isinstance(metrics, dict) or snr is None:
        return None
    basename = os.path.basename(source)
    if re.search(r'_test_ac\d|_ns77\d|_test_snr\d', basename):
        return None

    name = data.get('experiment') or data.get('experiment_name') or basename
    snr = int(float(snr))
    experiment = (name if name.endswith(('ns1', 'ns2', 'ns3')) or '_ns' in name
                  else basename.replace('_test.json', ''))

    match = re.match(r'rev_(\w+?)_(tpls|fcn)_(?:m512_)?s(\d+)', name)
    if match:
        dataset, model, seed = match.groups()
        ckpt = data.get('ckpt') or f'checkpoints/rev_{dataset}_{model}_m512_s{seed}'
        return _base_row(experiment, dataset, model, 'noise_ac', metrics, basename,
                         ckpt, seed=seed, eval_noise_db=snr, environment='rev2026-07')

    match = re.match(r'ta_(\w+?)_(hsi|cs)_s(\d+)_snr', name)
    if match:
        dataset, method, seed = match.groups()
        if seed == '42':  # legacy master CSV already owns this arm
            return None
        return _base_row(experiment, dataset, f'ta_{method}', 'ta_noise', metrics,
                         basename, f'checkpoints/ta_{dataset}_{method}_s{seed}',
                         seed=seed, eval_noise_db=snr, environment='rev2026-07')

    match = re.match(r'ta_(\w+?)_hsi_(naug20|naugmix)_snr', name)
    if match:
        dataset, tag = match.groups()
        return _base_row(experiment, dataset, f'ta_hsi_{tag}', 'ta_noise_naug',
                         metrics, basename, f'checkpoints/ta_{dataset}_hsi_{tag}',
                         seed=42, eval_noise_db=snr, environment='rev2026-07')
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    csv_in = argv[0] if argv else os.path.join(ROOT, 'results', 'master_results.csv')
    csv_out = argv[1] if len(argv) > 1 else os.path.join(ROOT, 'results', 'master_results_merged.csv')

    existing = []
    if os.path.exists(csv_in):
        with open(csv_in, newline='', encoding='utf-8') as handle:
            existing = list(csv.DictReader(handle))
    seen = {row['experiment'] for row in existing}

    paths = (
        glob.glob(os.path.join(ROOT, 'incoming', '**', '*.json'), recursive=True)
        + glob.glob(os.path.join(ROOT, 'checkpoints', '*', 'eval', '*.json'))
        + glob.glob(os.path.join(ROOT, 'results', '**', '*.json'), recursive=True)
        + glob.glob(os.path.join(ROOT, '_rev', 'results', '**', '*.json'), recursive=True)
    )
    new_rows = []
    for path in sorted(set(paths)):
        try:
            with open(path, encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        merged = noise_eval_row(data, path) or clean_eval_row(data, path)
        if merged is None or merged['experiment'] in seen:
            continue
        new_rows.append(merged)
        seen.add(merged['experiment'])

    all_rows = existing + new_rows
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    with open(csv_out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLS)
        writer.writeheader()
        for merged in all_rows:
            writer.writerow({column: merged.get(column, '') for column in COLS})

    families = {}
    for merged in new_rows:
        families[merged['family']] = families.get(merged['family'], 0) + 1
    print(f'existing={len(existing)}  new={len(new_rows)}  total={len(all_rows)}')
    print('new by family:', families)
    print('wrote', csv_out)


if __name__ == '__main__':
    main()
