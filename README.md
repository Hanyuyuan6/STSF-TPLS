# STSF-TPLS

> Segment straight from single-pixel measurements — no image reconstruction in the loop.

[![arXiv](https://img.shields.io/badge/arXiv-2607.22077-b31b1b.svg)](https://arxiv.org/abs/2607.22077)
[![License](https://img.shields.io/github/license/Hanyuyuan6/STSF-TPLS)](LICENSE)
[![CI](https://github.com/Hanyuyuan6/STSF-TPLS/actions/workflows/tests.yml/badge.svg)](https://github.com/Hanyuyuan6/STSF-TPLS/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%E2%80%932.7-red)](https://pytorch.org/)

A bidirectional GRU encodes the ordered 1D measurement sequence, a content-adaptive cross-attention
lift maps it onto a 2D feature grid, and a U-Net++ head decodes the mask — no intermediate image is
reconstructed. Official code for **"The Lift Spectrum: How Measurement-to-Space Adaptivity Shapes
Robustness in Image-Free Single-Pixel Sensing"** ([arXiv:2607.22077](https://arxiv.org/abs/2607.22077)).

## Quick start

Runs end-to-end on a fresh clone: no dataset to download (torchvision fetches MNIST), no GPU needed.

```bash
conda create -n sps python=3.10 && conda activate sps
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
                                     # the torch pins are +cu118 builds (they also run CPU-only);
                                     # for another CUDA build see pytorch.org/get-started/locally

python -m pytest test/ -q            # -> 23 passed, 1 skipped   (24 passed with a CUDA device;
                                     #    the GPU-parity test self-skips without one)
python -m scripts.train    --config configs/experiments/rev_mnist_tpls.yaml --seed 42 --epochs 1
python -m scripts.evaluate --ckpt_path checkpoints/rev_mnist_tpls_s42/best.pth --split test
# -> PA / mPA / mIoU / mDice and the foreground-only miou_fg / mdice_fg.
#    One epoch is a smoke test, not a converged model: the real schedules live in configs/experiments/.
```

Every command in this README assumes **the repo root as the working directory** and the
`python -m scripts.<name>` form; running a script by path (`python scripts/evaluate.py`) fails with
`ModuleNotFoundError: No module named 'src'`.

`requirements.txt` is pinned to the environment the reported evaluations were computed in
(torch 2.3.0+cu118; training was cross-checked on torch 2.7.0+cu128). Relax the pins to ranges for a
newer stack; run-to-run bit-exactness depends on the exact torch/CUDA build.

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
hf download hanyuyuan/STSF-TPLS-weights checkpoints/rev_carvana_tpls_s42/best.pth --local-dir .
python -m scripts.evaluate --ckpt_path checkpoints/rev_carvana_tpls_s42/best.pth --split test
```
[hanyuyuan/STSF-TPLS-weights](https://huggingface.co/hanyuyuan/STSF-TPLS-weights) lists every file with
its MD5 and the clean test foreground mIoU it reproduces. Training from scratch needs no download.

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
Run the full single-seed sweep (15 main + 6 lift + 6 reconstruction baselines):
```bash
bash scripts/run_all.sh                        # idempotent; skips finished runs
```

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
    --ckpt_path checkpoints/rev_carvana_tpls_s42/best.pth --save_path predictions/bucket_pred.png
```
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
- **Aggregating many runs.** `scripts/merge_results.py` collects `checkpoints/*/eval/*.json` and
  `results/**/*.json` into one row per evaluated run; `scripts/analyze_reversal.py` reads that table and
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
- **Checkpoint safety.** `predict_from_bucket.py` loads with `weights_only=True`; only
  load `.pth` / `.mat` files from sources you trust.

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
