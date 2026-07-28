# -*- coding: utf-8 -*-
"""Merge per-run eval JSONs into the results CSV (stdlib, NO torch).

Scans <repo>/checkpoints/<run>/eval/*.json (plus an optional <repo>/incoming staging dir),
maps each noise-eval JSON to a row of the 19-col schema, and dedups by `experiment`.
- image-free s43/s44 noise -> family=noise_ac
- TA seed noise (s43/s44)  -> family=ta_noise
- naug (naug20/naugmix)    -> family=ta_noise_naug (mechanism probe, not the deployed baseline)
Writes a STAGING file (default master_results_merged.csv); it never overwrites the input CSV.

Run (repo root):  python scripts/merge_results.py [csv_in] [csv_out]
  csv_in  default <repo>/results/master_results.csv — tolerated missing (from-scratch start)
  csv_out default <repo>/results/master_results_merged.csv

Provenance: this is the repo edition of the script the paper's tables were merged with. The row
schema, name regexes, legacy-protocol exclusions and dedup rules are unchanged; the deltas here
are this docstring, argv/ROOT path plumbing, missing-CSV tolerance, and makedirs for the output dir.
KNOWN ASYMMETRY (inherited, unchanged): TA rows with seed 42 are hard-skipped, because
TA-s42 run directories carry no `_s42` infix and name-level dedup cannot tell them apart
from rows already in csv_in. A from-scratch merge therefore lacks the TA s42 arm unless
those rows are supplied in csv_in. `analyze_reversal.py` prints the arms and seeds it
loaded — check that list rather than assuming the grid is complete.
"""
import csv, json, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (= scripts/..)
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'master_results.csv')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'master_results_merged.csv')
CKEV = os.path.join(ROOT, 'checkpoints')          # per-run eval JSONs live in <run>/eval/
OVN = os.path.join(ROOT, 'incoming')              # staging for future remote arrivals
RES = os.path.join(ROOT, 'results')               # where the README's own eval commands write
COLS = ['experiment','dataset','model','family','M','rate_pct','train_noise_db','eval_noise_db',
        'perm_seed','seed','env','miou_fg','mdice_fg','miou','mdice','pa','mpa','source','ckpt_path']

existing = list(csv.DictReader(open(CSV, newline='', encoding='utf-8'))) if os.path.exists(CSV) else []
seen = {r['experiment'] for r in existing}
new_rows = []

def pct(m, k):
    v = m.get(k)
    return '' if v is None else f'{v*100:.4f}'

def row(exp, ds, model, family, snr, seed, m, src, ckpt):
    return {'experiment': exp, 'dataset': ds, 'model': model, 'family': family, 'M': '512',
            'rate_pct': '3.125', 'train_noise_db': '', 'eval_noise_db': str(snr), 'perm_seed': '',
            'seed': str(seed), 'env': 'rev2026-07', 'miou_fg': pct(m,'miou_fg'), 'mdice_fg': pct(m,'mdice_fg'),
            'miou': pct(m,'miou'), 'mdice': pct(m,'mdice'), 'pa': pct(m,'pa'), 'mpa': pct(m,'mpa'),
            'source': src, 'ckpt_path': ckpt}

# results/ is included because the README's own noise commands write there
# (`evaluate.py --out_json results/*.json`, `ta_noise_eval.py` -> results/ta_noise/).
# Entries without a `noise_snr_db` key are skipped below, so unrelated JSONs are inert.
_paths = (glob.glob(os.path.join(OVN, '**', '*.json'), recursive=True)
          + glob.glob(os.path.join(CKEV, '*', 'eval', '*.json'))
          + glob.glob(os.path.join(RES, '**', '*.json'), recursive=True))
for f in _paths:
    d = json.load(open(f)); m = d.get('metrics', {})
    name = d.get('experiment') or d.get('experiment_name') or os.path.basename(f)
    snr = d.get('noise_snr_db'); src = os.path.basename(f)
    if snr is None:
        continue
    # legacy-protocol exclusions (June-era full-ref noise `_test_acXX`, ns777+ extra-seed batch,
    # old `_test_snrXX` full-ref): archived in checkpoints/<run>/eval/ but NEVER merged — the live
    # noise axis is the ac-referenced ns1-3 protocol only (family noise_ac / ta_noise).
    if re.search(r'_test_ac\d|_ns77\d|_test_snr\d', os.path.basename(f)):
        continue
    snr = int(float(snr)); exp = name if name.endswith(('ns1','ns2','ns3')) or '_ns' in name else os.path.basename(f).replace('_test.json','')
    if exp in seen:
        continue
    # image-free: rev_{ds}_{tpls|fcn}[_m512]_s{seed}  (wbc configs carry no m512 in experiment_name)
    mo = re.match(r'rev_(\w+?)_(tpls|fcn)_(?:m512_)?s(\d+)', name)
    if mo:
        ds, model, seed = mo.group(1), mo.group(2), mo.group(3)
        ckpt = d.get('ckpt') or f'checkpoints/rev_{ds}_{model}_m512_s{seed}'
        new_rows.append(row(exp, ds, model, 'noise_ac', snr, seed, m, src, ckpt)); seen.add(exp); continue
    # TA seed: ta_{ds}_{hsi|cs}_s{seed}_snr…  (skip s42: already in master CSV -> avoid double-count)
    mo = re.match(r'ta_(\w+?)_(hsi|cs)_s(\d+)_snr', name)
    if mo:
        ds, meth, seed = mo.group(1), mo.group(2), mo.group(3)
        if seed == '42':
            continue
        new_rows.append(row(exp, ds, f'ta_{meth}', 'ta_noise', snr, seed, m, src, f'checkpoints/ta_{ds}_{meth}_s{seed}')); seen.add(exp); continue
    # naug: ta_{ds}_hsi_{naug20|naugmix}_snr…
    mo = re.match(r'ta_(\w+?)_hsi_(naug20|naugmix)_snr', name)
    if mo:
        ds, tag = mo.group(1), mo.group(2)
        new_rows.append(row(exp, ds, f'ta_hsi_{tag}', 'ta_noise_naug', snr, 42, m, src, f'checkpoints/ta_{ds}_hsi_{tag}')); seen.add(exp); continue

allrows = existing + new_rows
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in allrows:
        w.writerow({c: r.get(c, '') for c in COLS})
print(f'existing={len(existing)}  new={len(new_rows)}  total={len(allrows)}')
fam = {}
for r in new_rows: fam[r['family']] = fam.get(r['family'], 0) + 1
print('new by family:', fam)
print('wrote', OUT)
