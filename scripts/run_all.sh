#!/usr/bin/env bash
# Full re-run (single seed 42, attn-STSF + GPU-bucket): A rev 15 + C WBC lift ablation 6 + B reconstruction baselines.
# Idempotent (skipped when the result json exists); traditional runs first per dataset (recon depends on it).
set -u
cd "$(dirname "$0")/.." || exit 1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PWD
SEED=42
mkdir -p _rev/results _rev/logs

train_eval () {   # $1 config-name  $2 test-json  $3 log  $4 also_val(1/0)
  local CFG=$1 TJ=$2 LG=$3 VALF=$4
  local CK=checkpoints/${CFG}_s${SEED}/best.pth
  if [ -f "$TJ" ]; then echo "[$CFG] done, skip"; return; fi
  if [ -f "$CK" ]; then
    echo "[$CFG] ckpt exists, skip TRAIN -> re-eval (a prior train succeeded but eval left no result json)"
  else
    echo "===== [$CFG] TRAIN $(date +%H:%M:%S) ====="
    python scripts/train.py --config configs/experiments/${CFG}.yaml --seed $SEED > "$LG" 2>&1
    if [ ! -f "$CK" ]; then echo "[$CFG] !! NO CKPT"; tail -6 "$LG"; return; fi
  fi
  if [ "$VALF" = "1" ]; then
    python scripts/evaluate.py --ckpt_path "$CK" --split val --out_json "${TJ%_test.json}_val.json" >> "$LG" 2>&1
  fi
  python scripts/evaluate.py --ckpt_path "$CK" --split test --out_json "$TJ" >> "$LG" 2>&1
  echo "[$CFG] -> $TJ"; tail -1 "$LG"
}

# ===== A: rev main comparison 15 (traditional first, feeds recon) =====
for CFG in rev_carvana_traditional rev_carvana_fcn rev_carvana_no_aux rev_carvana_fixed rev_carvana_tpls \
           rev_mnist_traditional   rev_mnist_fcn   rev_mnist_no_aux   rev_mnist_fixed   rev_mnist_tpls \
           rev_wbc_traditional     rev_wbc_fcn     rev_wbc_no_aux     rev_wbc_fixed     rev_wbc_tpls; do
  train_eval "$CFG" "_rev/results/${CFG}_s${SEED}.json" "_rev/logs/${CFG}.log" 0
done

# ===== C: WBC lift ablation 6 (val+test) =====
for L in gru srconv attn inr mamba kan; do
  CFG=lift_wbc_${L}
  train_eval "$CFG" "_rev/results/${CFG}_test.json" "_rev/logs/${CFG}.log" 1
done

# ===== B: reconstruction baselines (HSI=tradgi / CS=admm-l1, depends on the traditional ckpt; saves only the num_vis panels, no save_all) =====
for ds in carvana mnist wbc; do
  CK=checkpoints/rev_${ds}_traditional_s${SEED}/best.pth
  CFGR=configs/experiments/rev_${ds}_traditional.yaml
  [ -f "$CK" ] || { echo "recon skip $ds: no traditional ckpt"; continue; }
  for m in tradgi admm-l1; do
    RJ=_rev/results/recon_${ds}_${m}_s${SEED}.json
    [ -f "$RJ" ] && { echo "recon skip $ds/$m"; continue; }
    EX=""; [ "$m" = "admm-l1" ] && EX="--reg_weight 0.01 --rho 1.0 --steps 100"
    echo "===== RECON $ds/$m $(date +%H:%M:%S) ====="
    python -m scripts.reconstruct_eval --config "$CFGR" --ckpt_path "$CK" --method "$m" \
      --sampling_rate 0.03125 --split test --num_vis 6 \
      --save_vis_dir recon_vis_${m}_${ds}/ --out_json "$RJ" $EX > _rev/logs/recon_${ds}_${m}.log 2>&1 \
      && echo "recon $ds/$m -> $RJ" || { echo "recon $ds/$m FAIL"; tail -5 _rev/logs/recon_${ds}_${m}.log; }
  done
done

echo "RUN_ALL_DONE $(date +%H:%M:%S)"
