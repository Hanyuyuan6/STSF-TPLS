# -*- coding: utf-8 -*-
"""Multi-seed reversal analysis (stdlib only, NO torch).

Aggregates the measurement-noise grid into the paper's reversal table: per (dataset, SNR),
STSF image-free mean/std/min across seeds vs the strongest CLEAN-TRAINED task-adapted arm
(mean/std/max), the mean margin, and the worst-case seed pairing (STSF min - TA max).
naug arms are a mechanism probe (noise-augmented), NOT the deployed baseline -> excluded
from "strongest TA" by design.

Input: the results CSV produced by scripts/merge_results.py (families noise_ac / ta_noise).
Run (repo root):  python scripts/analyze_reversal.py [path/to/master_results.csv]
                  (default: <repo>/results/master_results.csv)

Provenance: this is the repo edition of the script the paper's reversal analysis was run with.
The aggregation logic, family whitelist, blacklist rationale and print format are unchanged;
the deltas here are this docstring, `import sys`, and the ROOT/CSV path plumbing below
(argv override + repo-root default).
"""
import json, csv, re, glob, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (= scripts/..)
OVN = os.path.join(ROOT, 'overnight_results')
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'master_results.csv')
rows = []  # each: dict(dataset, arm, seed, snr, ns, fg)

def add(dataset, arm, seed, snr, ns, fg_pct):
    # fg_pct is ALWAYS already on the percent (0-100) scale. NO <=1 ->*100 heuristic:
    # a genuine sub-1% collapse (e.g. 0.69% fg-mIoU at 10 dB) must stay 0.69, not become 69.
    # JSON callers (fractions) convert *100 before calling; CSV callers pass percent verbatim.
    if fg_pct is None or dataset is None or arm is None or fg_pct == '': return
    rows.append(dict(dataset=dataset, arm=arm, seed=int(seed), snr=int(float(snr)),
                     ns=(int(ns) if ns is not None else 0), fg=float(fg_pct)))

# --- 1) JSONs (SKIP by default: the merged CSV is the single source of truth, and reading the
#        JSONs on top of it would double-count s43/s44. Set USE_JSON=1 to also scan OVN, an
#        optional local drop-box for eval JSONs that no script in this repo writes.) ---
for f in (glob.glob(os.path.join(OVN, '**', '*.json'), recursive=True) if os.environ.get('USE_JSON') else []):
    d = json.load(open(f))
    m = d.get('metrics', {}); fg = m.get('miou_fg')
    fg = fg * 100.0 if fg is not None else None   # JSON stores fractions -> convert to percent
    name = d.get('experiment') or d.get('experiment_name') or os.path.basename(f)
    snr = d.get('noise_snr_db'); ns = d.get('noise_seed')
    # image-free: rev_{ds}_{tpls|fcn}_m512_s{seed}
    mo = re.match(
        r'^rev_([A-Za-z0-9][A-Za-z0-9-]*)_(tpls|fcn)_m512_s(\d+)(?:_|$)',
        name,
    )
    if mo:
        ds, mdl, seed = mo.group(1), mo.group(2), mo.group(3)
        add(ds, 'stsf' if mdl == 'tpls' else 'spifs', seed, snr, ns, fg); continue
    # TA seed: ta_{ds}_{hsi|cs}_s{seed}
    mo = re.match(
        r'^ta_([A-Za-z0-9][A-Za-z0-9-]*)_(hsi|cs)_s(\d+)(?:_|$)',
        name,
    )
    if mo:
        ds, meth, seed = mo.group(1), mo.group(2), mo.group(3)
        add(ds, f'ta_{meth}', seed, snr, ns, fg); continue
    # TA naug: ta_{ds}_hsi_{naug20|naugmix}
    mo = re.match(
        r'^ta_([A-Za-z0-9][A-Za-z0-9-]*)_hsi_(naug20|naugmix)(?:_|$)',
        name,
    )
    if mo:
        ds, tag = mo.group(1), mo.group(2)
        add(ds, f'ta_{tag}', 42, snr, ns, fg); continue

# --- 2) master_results.csv (s42): ONLY AC-ref noise families. Exclude blacklisted
#        'noise_eval' (full-ref 47.1 cliff artifact) + deprecated 'train_noisy'. ---
if os.path.exists(CSV):
    for r in csv.DictReader(open(CSV)):
        fam = r.get('family', '').strip()
        if fam not in ('noise_ac', 'ta_noise'):
            continue
        exp = r['experiment']; snr = r.get('eval_noise_db', '').strip(); fg = r.get('miou_fg', '').strip()
        if not fg or not snr: continue
        seed = r.get('seed', '42').strip() or '42'
        mo = re.search(r'_ns(\d+)', exp); ns = mo.group(1) if mo else None
        if fam == 'noise_ac':
            # `_m512` is optional: the table this script was developed against came from the `rev_*_m512_*` run
            # family, but run_all.sh / the README train the same 3.13% models under the
            # plain `rev_<ds>_<model>` name. Requiring the suffix silently dropped every
            # image-free row a fresh clone produces (empty table, exit 0).
            m2 = re.match(
                r'^rev_([A-Za-z0-9][A-Za-z0-9-]*)_(tpls|fcn)(?:_m512)?(?:_|$)',
                exp,
            )
            if m2: add(m2.group(1), 'stsf' if m2.group(2) == 'tpls' else 'spifs', seed, snr, ns, fg)
        else:  # ta_noise
            m3 = re.match(
                r'^ta_([A-Za-z0-9][A-Za-z0-9-]*)_(hsi|cs)(?:_|$)',
                exp,
            )
            if m3: add(m3.group(1), f'ta_{m3.group(2)}', seed, snr, ns, fg)

# --- guard: no input resolved (a fresh clone ships neither the overnight JSONs nor the CSV) ---
if not rows:
    sys.stderr.write(
        "ERROR: no reversal records loaded — nothing to aggregate.\n"
        f"  CSV looked for: {CSV} (exists={os.path.exists(CSV)})\n"
        f"  optional USE_JSON drop-box: {OVN} (exists={os.path.isdir(OVN)})\n"
        f"  per-run eval JSONs are written under checkpoints/<run>/eval/ by evaluate.py\n"
        "  Metric tables are not distributed with the repo — build one from your own runs:\n"
        "      python scripts/merge_results.py\n"
        "  (aggregates checkpoints/<run>/eval/*.json and results/**/*.json into\n"
        "   results/master_results_merged.csv), then point this script at it:\n"
        "      python scripts/analyze_reversal.py results/master_results_merged.csv\n")
    sys.exit(1)

# --- 3) reversal table ---
def sel(ds, arm, snr):
    return [r['fg'] for r in rows if r['dataset'] == ds and r['arm'] == arm and r['snr'] == snr]

datasets = sorted(set(r['dataset'] for r in rows))
print(f"loaded {len(rows)} records | datasets={datasets}")
print(f"arms present: {sorted(set(r['arm'] for r in rows))}\n")
for ds in datasets:
    print(f"===== {ds.upper()} =====")
    print(f"{'SNR':>4} | {'STSF n/mean/std/min':>26} | {'strongest-TA n/mean/std/max':>30} | {'margin':>7} | {'worst':>7}")
    for snr in [40, 30, 20, 15, 10]:
        s = sel(ds, 'stsf', snr)
        if not s: continue
        s_seeds = sorted(set(r['seed'] for r in rows if r['dataset']==ds and r['arm']=='stsf' and r['snr']==snr))
        # strongest TA = the strongest CLEAN-TRAINED TA (hsi/cs) at this snr.
        # naug arms are a mechanism probe (noise-augmented), NOT the deployed baseline -> excluded.
        ta_arms = [a for a in set(r['arm'] for r in rows if r['dataset']==ds and r['snr']==snr)
                   if a.startswith('ta_') and 'naug' not in a]
        best = None
        for a in ta_arms:
            v = sel(ds, a, snr)
            if v and (best is None or st.mean(v) > st.mean(best[1])):
                best = (a, v)
        smean, sstd, smin = st.mean(s), (st.pstdev(s) if len(s)>1 else 0), min(s)
        if best:
            a, tv = best; tmean, tstd, tmax = st.mean(tv), (st.pstdev(tv) if len(tv)>1 else 0), max(tv)
            margin, worst = smean - tmean, smin - tmax
            # 4dp on the MEANS, not 2dp: a 2dp console is a double-rounding vector. Reading a
            # 1dp figure off a 2dp print rounds twice and can land a full step away from the
            # full-precision mean. Take reported numbers from the 4dp column, never from a 2dp one.
            print(f"{snr:>4} | seeds{s_seeds} {smean:8.4f}/{sstd:4.2f}/{smin:6.2f} | {a:>10} {tmean:8.4f}/{tstd:4.2f}/{tmax:6.2f} | {margin:+8.4f} | {worst:+6.2f}")
        else:
            print(f"{snr:>4} | seeds{s_seeds} {smean:8.4f}/{sstd:4.2f}/{smin:6.2f} | (no TA yet)")
    print()
