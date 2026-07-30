# STSF-TPLS

> Segment straight from single-pixel measurements — no image reconstruction in the loop.

[![arXiv](https://img.shields.io/badge/arXiv-2607.22077-b31b1b.svg)](https://arxiv.org/abs/2607.22077)
[![License](https://img.shields.io/github/license/Hanyuyuan6/STSF-TPLS)](LICENSE)
[![CI](https://github.com/Hanyuyuan6/STSF-TPLS/actions/workflows/tests.yml/badge.svg)](https://github.com/Hanyuyuan6/STSF-TPLS/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.13.0-red)](https://pytorch.org/)

A bidirectional GRU encodes the ordered 1D measurement sequence, a content-adaptive cross-attention
lift maps it onto a 2D feature grid, and a U-Net++ head decodes the mask — no intermediate image is
reconstructed. Official code for **"The Lift Spectrum: How Measurement-to-Space Adaptivity Shapes
Robustness in Image-Free Single-Pixel Sensing"** ([arXiv:2607.22077](https://arxiv.org/abs/2607.22077)).

## Quick start

Runs end-to-end on a fresh clone: no dataset to download (torchvision fetches MNIST), no GPU needed.

```bash
conda create -n sps python=3.10 && conda activate sps
pip install -r requirements.txt       # reviewed exact direct pins; transitive resolution is not hash-locked

python -m pytest test/ -q            # the GPU-parity test self-skips without CUDA
python -m scripts.train    --config configs/experiments/rev_mnist_tpls.yaml --seed 42 --epochs 1
python -m scripts.evaluate --ckpt_path checkpoints/rev_mnist_tpls_s42/best.pth --split test
# -> PA / mPA / mIoU / mDice and the foreground-only miou_fg / mdice_fg.
#    One epoch is a smoke test, not a converged model: the real schedules live in configs/experiments/.
```

Every command in this README assumes **the repo root as the working directory** and the
`python -m scripts.<name>` form; running a script by path (`python scripts/evaluate.py`) fails with
`ModuleNotFoundError: No module named 'src'`.

`requirements.txt` pins every direct dependency in the current reviewed Python 3.10 environment
(PyTorch 2.13.0); transitive packages are resolver-selected rather than locked by hashes. For provenance,
the original paper evaluation used PyTorch 2.3.0+cu118 and training was cross-checked on 2.7.0+cu128.
Those historical, known-vulnerable versions are recorded only to explain the reported artifacts:
they are not recommended installation targets, and their checkpoints must be treated as trusted,
local inputs. Run-to-run bit-exactness still depends on the exact PyTorch/CUDA build.

## Models (`src/models/`)

| Class | Structure | Config prefix |
|---|---|---|
| `GRUUNetPP` + `AttnLift` | bi-GRU → cross-attention lift → U-Net++ | `rev_*_tpls` / `*_no_aux` / `*_fixed` |
| `FCNUNetPP` | encoder → FSRCNN → U-Net++ | `rev_*_fcn` |
| `BaselineUNetPP` | image-domain U-Net++ (consumes an image, not a bucket) | `rev_*_traditional` / `ta_*` |
| `LiftUNetPP` | shared GRU front end → swappable 1D→2D lift (`gru`/`srconv`/`attn`/`inr`/`mamba`/`kan`) → shared U-Net++ tail | `lift_wbc_*` |

## Datasets
Three datasets are used: **Carvana** (binary vehicle masks), **MNIST** (binary digit
foreground), and **WBC** (3-class white-blood-cell). MNIST downloads automatically via
torchvision. Download the other two yourself:

- **Carvana** — [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)
  (`train.zip` + `train_masks.zip`: `*.jpg` images with `*_mask.gif` masks).
- **WBC** — [zxaoyou/segmentation_WBC](https://github.com/zxaoyou/segmentation_WBC)
  (paired `.bmp` images and `.png` masks; Zheng et al., *Micron* 107:55–71, 2018).

Place the archives under `data_rev/<dataset>/raw/` (the experiment configs read `data_rev/`),
then build the group-disjoint / sample-disjoint train/val/test splits:
```bash
python -m scripts.prepare_seg_datasets --dataset carvana --data_root data_rev
python -m scripts.prepare_seg_datasets --dataset wbc     --data_root data_rev
```
Carvana is split by vehicle ID (seeded, with a hard disjointness assertion) so the same car
never appears in two splits.

## Pretrained weights
The ten STSF+TPLS checkpoints behind the paper's main comparison (three datasets × three seeds, plus
the optical-bench checkpoint) are on the Hugging Face Hub as inference-only exports — optimizer and
scheduler state stripped, 147.6 MB each, loadable under `weights_only=True`:
```bash
pip install huggingface_hub          # provides the `hf` CLI; kept out of requirements.txt
hf download hanyuyuan/STSF-TPLS-weights checkpoints/rev_carvana_tpls_s42/best.pth \
    --revision v1.0.0 --local-dir .
python -m scripts.evaluate --ckpt_path checkpoints/rev_carvana_tpls_s42/best.pth --split test
```
[hanyuyuan/STSF-TPLS-weights](https://huggingface.co/hanyuyuan/STSF-TPLS-weights) lists every file with
its exact byte count, Git LFS SHA-256 object ID, legacy MD5, and clean test foreground mIoU.
Training from scratch needs no download.

## Running the experiments

### Train
Each experiment is a single YAML in `configs/experiments/`. The checkpoint name is the
config's `experiment_name` plus the seed suffix (e.g. `rev_carvana_tpls_s42`).
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8        # determinism
python -m scripts.train --config configs/experiments/rev_carvana_tpls.yaml --seed 42
```
For a quick end-to-end smoke run that needs no manual download, use MNIST (torchvision
fetches it automatically) and cap the schedule with `--epochs`:
```bash
python -m scripts.train --config configs/experiments/rev_mnist_tpls.yaml --seed 42 --epochs 1
python -m scripts.evaluate --ckpt_path checkpoints/rev_mnist_tpls_s42/best.pth --split test
```
Run the released clean-config orchestrator. Its default is an explicit one-seed smoke sweep;
`MODE=full` runs seeds 42/43/44. Neither mode silently claims the separate TA, noise, rate, or
mechanism sweeps:
```bash
MODE=smoke RUN_ID=smoke01 bash scripts/run_all.sh  # one seed; integration/smoke scope
MODE=full  RUN_ID=repro01 bash scripts/run_all.sh  # 3 seeds; released clean configs only
```
Each safe `RUN_ID` gets unique checkpoint, TensorBoard, result, and log paths; any pre-existing target
fails instead of reusing a stale `best.pth`. The script validates the exact JSON inventory plus every
artifact's split, dataset, seed, model, experiment identity, sample count and finite metrics, then writes
a SHA-256 manifest under `_rev/results/run_all/<RUN_ID>/`; stale, extra, truncated, or mislabeled JSON fails.

### Evaluate (test split, foreground metrics)
`evaluate.py` reports PA / mPA / mIoU / mDice **and** foreground-only `miou_fg` / `mdice_fg`
(the background class is excluded from the per-class mean). Foreground occupies a small fraction
of each scene, so background IoU saturates and inflates the class-averaged scores — read the
foreground-only pair.
```bash
python -m scripts.evaluate --ckpt_path checkpoints/rev_carvana_tpls_s42/best.pth \
    --split test --out_json results/rev_carvana_tpls.json \
    --save_vis_dir eval_vis/rev_carvana_tpls/
```

### Reconstruct-then-segment baselines (HSI / CS)
Classical reconstruction (traditional ghost imaging or ADMM-L1 compressed sensing)
followed by the per-dataset image-domain segmenter:
```bash
python -m scripts.reconstruct_eval --config configs/experiments/rev_carvana_traditional.yaml \
    --ckpt_path checkpoints/rev_carvana_traditional_s42/best.pth \
    --method tradgi --sampling_rate 0.03125 --split test --save_all --save_vis_dir recon_vis_GI/
python -m scripts.reconstruct_eval --config configs/experiments/rev_carvana_traditional.yaml \
    --ckpt_path checkpoints/rev_carvana_traditional_s42/best.pth \
    --method admm-l1 --sampling_rate 0.03125 --split test --save_all --save_vis_dir recon_vis_ADMM/ \
    --reg_weight 0.01 --rho 1.0 --steps 100
```

### Measurement-noise evaluation
Both families are evaluated on measurements corrupted at the **same SNR** under the same AC-coupled
bucket calibration (`sigma = std(raw[:, 1:]) * 10^(-SNR/20)`; row 0 is the DC coefficient and is
excluded). Each arm draws its own noise realization: the arms differ in batch size, and on CUDA the
Philox offset advances per kernel launch, so a shared `--noise_seed` does not give a shared draw.
The protocol is matched-SNR; average over 3 noise seeds x 3 training seeds to get a comparable number.

```bash
# image-free arms (STSF / SPIFS) — noise injected in the measurement domain at eval time
python -m scripts.evaluate --ckpt_path checkpoints/rev_mnist_tpls_s42/best.pth \
    --split test --noise_snr_db 20 --noise_ref ac --noise_seed 1 --out_json results/stsf_snr20.json

# reconstruct-then-segment arm (task-adapted) — same SNR and calibration, then reconstruct, then segment.
# Defaults sweep SNR in {40,30,20,10} x 3 noise seeds x 3 training seeds.
SEEDS="42 43 44" DATASETS="carvana mnist wbc" METHODS="hsi cs" python -m scripts.ta_noise_eval
```

The mechanism diagnostic does not need a trained model. It applies one identical, seeded noise
realization to both lifts and reports the linear relative perturbations, the Eq. (7) prediction, and
the actual shift after each arm's min-max normalization, with per-sample values retained in JSON:

```bash
python -m scripts.noise_amplification --dataset mnist --snr_db 20 \
    --output results/noise_amplification_mnist_20db.json
```
With no seed override this runs the complete selected split for noise seeds 1/2/3, applies the
deployed 8-bit reconstruction round-trip, and retains both per-seed and per-sample records.
`--max_samples` deliberately changes it into a diagnostic subset; `--noise_seed` is a documented
single-seed compatibility mode. Eq. (7) is reported as the paper's first-order orthogonal-row
approximation alongside the realized linear and post-normalization shifts.

### TPLS gradient-dynamics diagnostic

Gradient instrumentation is opt-in and measures the two **unweighted** loss gradients before the
ordinary weighted optimizer step. It writes cosine similarity, `Pr[cos<0]`, both norms, the raw
auxiliary/task magnitude ratio, gradient-magnitude similarity, and the runtime-weighted effective
pull. The released Supplement S6 protocol uses
MNIST, seed 42, nine epochs, 40 optimizer steps per epoch, and five instrumented repeats:

```bash
python -m scripts.run_gradient_protocol \
    --config configs/experiments/rev_mnist_tpls.yaml \
    --output_dir results/gradient_diagnostics
```

The runner refuses to overwrite its reserved output files, verifies exactly 360 records per repeat,
and preserves the 0.3/0.3/0.4 schedule as exactly 108/108/144 optimizer steps rather than rounding
whole epochs. Every reported statistic must be finite at every step; missing/undefined gradients fail
the run rather than being dropped. It averages within each run/phase before assigning equal weight to the five repeats.
`protocol.json` records the commands, environment, Git commit, and a working-tree snapshot hash that
also covers untracked files; `summary.json` contains per-run means and the across-run ranges.

The task-adapted segmenters are not trained on live reconstructions. `recon_dump.py` reconstructs the
whole split once and writes the results to disk as an ordinary image folder
(`<out_root>/<method>_<dataset>/<split>/{images,masks}/*.png`, grayscale 128×128); the `ta_*` configs
then train on those files like any image dataset. Add `--noise_snr_db` / `--noise_mixed` to produce the
noise-augmented variants. To score an off-the-shelf segmenter's predicted masks against an already
written-out set, use `scripts/ta_infer_20db.py` (it writes mask PNGs; it computes no metric of its own).

### Inference from real hardware buckets
Real-hardware bucket `.mat` files are not shipped (they are gitignored); supply your own
measurement file, or capture one with the acquisition setup described in the paper.
```bash
python -m scripts.predict_from_bucket --bucket_path path/to/your_bucket.mat \
    --mat_key d_B --ckpt_path checkpoints/rev_carvana_tpls_s42/best.pth \
    --save_path predictions/bucket_pred.png
```
If a `.mat` file contains more than one numeric vector of the requested length, inference fails
closed until `--mat_key` identifies the intended bucket (the acquisition files normally use `d_B`).
`predict_reconstruct.py` applies the same rule to both `.mat` and `.npz` archives.
`scripts/realtest_eval.py` scores STSF / SPIFS / TA-HSI on the same capture, plus the run-to-run
agreement, against a clean simulated upper bound. It expects the run names
`rev_mnist_tpls_m512_s42`, `rev_mnist_fcn_m512_s42` and `ta_mnist_hsi_s42`, so train
`rev_mnist_{tpls,fcn}_m512.yaml` first (see the `_m512` note below) — `run_all.sh` alone does not
produce those two.
```bash
python -m scripts.realtest_eval --real_dir path/to/captures --out_dir realtest_out
```

## Logging
Training metrics, losses, and sample images are written with **TensorBoard** to `runs/`:
```bash
tensorboard --logdir runs
```

## Notes
- **Acquisition.** Every config uses the same fixed **natural-order (Sylvester)** Hadamard basis —
  not a sequency-ordered one — at 3.125 % sampling (M = 512 of 128×128).
- **WBC loss.** The three-class WBC objective uses Dice+CE (0.5/0.5) with no class
  weighting, so background dominates the CE term and the foreground scores your run
  prints are lower than a class-weighted objective would give.
- **Seeds.** Seeds are not config keys: re-run any config under another seed with `--seed 43`,
  and the checkpoint directory picks up the `_s43` suffix automatically. One run gives one
  seed — average over 42/43/44 before comparing against any multi-seed number.
- **Latency.** `scripts/measure_latency.py` times a trained checkpoint at batch 1 (CUDA-synchronized,
  warmup included; `--device cuda|cpu`). `scripts/bench_latency.py` compares the image-free and
  reconstruct-then-segment pipelines end-to-end from random init (no checkpoint needed) but is
  **CUDA-only** — it has no CPU fallback.
- **Aggregating many runs.** `scripts/merge_results.py` collects clean and noise JSONs from
  `checkpoints/*/eval/`, `results/`, `_rev/results/`, and `incoming/` into one row per evaluated run;
  `scripts/analyze_reversal.py` reads that table and
  prints, per dataset and SNR, the image-free arm against the strongest reconstruct-then-segment arm.
  It also prints the arms and seeds it actually loaded — check that list, a from-scratch merge skips TA
  runs at seed 42 (their directory names carry no `_s42` infix; see the docstring in `merge_results.py`).
- **Configs outside `run_all.sh`.** `configs/experiments/` also ships these study families. They are
  separate studies, so run them individually with
  `python -m scripts.train --config configs/experiments/<name>.yaml --seed 42`.

  | Study | Configs |
  |---|---|
  | Sampling-rate sweep | `rev_{carvana,mnist}_{fcn,tpls}_m{64,128,256,512,1024,2048}`, `rev_wbc_{fcn,tpls}_m{64,128,256,1024,2048}` |
  | Noise-augmented training | `rev_carvana_{fcn,tpls}_tn{20,30}` |
  | Reconstruct-then-segment baseline | `ta_{carvana,mnist,wbc}_{cs,hsi}`, rate points `…_m{64,128,256,1024,2048}` |
  | Noise-augmented TA probes | `ta_mnist_hsi_naug{20,mix}` |
  | Acquisition-order ablation | `rev_mnist_tpls_perm123` |

  WBC's 3.13 % point is the base `rev_wbc_*` config itself. For carvana/mnist the `_m512` files are
  **deliberate aliases** of the base configs — identical except for `experiment_name` — so that the
  3.13 % point also exists under the `_m512` run name that `realtest_eval.py`, `merge_results.py` and
  `analyze_reversal.py` recognise. They are not redundant copies; don't delete them.
- **Scope.** This repo covers the segmentation experiments and emits no PSNR/SSIM. The reconstruction
  probe that selects the temporal encoder (PSNR/SSIM over eight encoder architectures) lives in a
  separate benchmark repository,
  [Hanyuyuan6/Comparison-of-SPI-DLs](https://github.com/Hanyuyuan6/Comparison-of-SPI-DLs).
- **Task-adapted configs read reconstructions from disk.** Each `ta_*` config trains a
  `BaselineUNetPP` on the PNG folder `recon_dump.py` writes, so generate the matching set before
  training — the output root encodes the variant:
  ```bash
  # the 3.13% baseline -> data_recon/
  python -m scripts.recon_dump --dataset mnist --method hsi --split train   # (repeat: val, test)
  # a rate point -> data_recon_m64/
  python -m scripts.recon_dump --dataset mnist --method hsi --split train --bucket_size 64 \
      --out_root data_recon_m64
  # the noise-augmented probes -> data_recon_naug20/ and data_recon_naugmix/
  python -m scripts.recon_dump --dataset mnist --method hsi --split train --noise_snr_db 20 \
      --out_root data_recon_naug20
  python -m scripts.recon_dump --dataset mnist --method hsi --split train --noise_mixed \
      --out_root data_recon_naugmix
  ```
  Every dump publishes `_dump_meta.json` before writing PNGs. Its generation signature covers the
  source sample order, sensing/reconstruction parameters, noise policy and execution backend. A
  completed dump also records the exact image/mask stem set plus every PNG's SHA-256. Training rejects
  partial, truncated, tampered, extra-file, legacy, and mismatched-signature reconstruction roots;
  select a new `--out_root` after reviewing old data. Released `ta_*` experiments require this manifest
  explicitly even if the root is renamed; custom experiment names should set
  `data.require_reconstruction_manifest: true` when they consume a reconstruction dump.
- **Checkpoint safety.** Checkpoint-consuming CLIs use `weights_only=True` and fail closed by default.
  `--allow_unsafe_pickle` is an explicit escape hatch for a trusted legacy checkpoint and can execute
  pickle payloads. `weights_only=True` is risk reduction, not a security sandbox on affected PyTorch
  versions. Treat all `.pth` and hardware `.mat` files as untrusted unless you verified their source;
  use the current reviewed environment rather than the historical paper environment for new work.

## Citation
If you use this code, please cite the preprint
[arXiv:2607.22077](https://arxiv.org/abs/2607.22077):

```bibtex
@article{han2026liftspectrum,
  title         = {The Lift Spectrum: How Measurement-to-Space Adaptivity Shapes
                   Robustness in Image-Free Single-Pixel Sensing},
  author        = {Han, Yuyuan and Li, Jingwei and Qiu, Long and Wang, Chong and
                   Hao, Wenxuan and Han, Jiangyu and Yao, Xinyu and He, Yuchen and
                   Chen, Hui and Liu, Jianbin and Zheng, Huaibin},
  journal       = {arXiv preprint arXiv:2607.22077},
  year          = {2026},
  eprint        = {2607.22077},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV}
}
```

`CITATION.cff` carries the same entry, so GitHub's *Cite this repository* button exports it directly.

## License
Released under the MIT License — see [LICENSE](LICENSE). Copyright (c) 2025 Yuyuan Han.
