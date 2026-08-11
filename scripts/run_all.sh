#!/usr/bin/env bash
# Released clean-experiment orchestrator. This is not the complete paper pipeline:
# TA, noise, sampling-rate, and mechanism sweeps have separate documented commands.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$PWD
PYTHON_BIN=${PYTHON_BIN:-python}

MODE=${MODE:-smoke}
case "$MODE" in
  smoke)
    SEEDS=${SEEDS:-42}
    echo "MODE=smoke: one seed, released clean configs only; NOT a full paper reproduction."
    ;;
  full)
    SEEDS=${SEEDS:-"42 43 44"}
    echo "MODE=full: seeds 42/43/44 for released clean configs."
    echo "This still excludes TA/noise/rate/mechanism sweeps; see README for those commands."
    ;;
  *)
    echo "Unknown MODE=$MODE (expected smoke or full)" >&2
    exit 2
    ;;
esac
read -r -a SEED_ARRAY <<< "$SEEDS"
if [ "${#SEED_ARRAY[@]}" -eq 0 ]; then
  echo "SEEDS must contain at least one integer" >&2
  exit 2
fi
for seed_value in "${SEED_ARRAY[@]}"; do
  [[ "$seed_value" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed_value" >&2; exit 2; }
done
if [ "$MODE" = "smoke" ] && [ "${#SEED_ARRAY[@]}" -ne 1 ]; then
  echo "MODE=smoke requires exactly one seed" >&2
  exit 2
fi
if [ "$MODE" = "full" ] && [ "${SEED_ARRAY[*]}" != "42 43 44" ]; then
  echo "MODE=full is the released clean protocol and requires SEEDS='42 43 44'" >&2
  exit 2
fi

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RUN_ID: $RUN_ID" >&2; exit 2; }
[[ "$RUN_ID" != "." && "$RUN_ID" != ".." ]] || { echo "Invalid RUN_ID: $RUN_ID" >&2; exit 2; }
RESULT_ROOT=${RESULT_ROOT:-_rev/results/run_all/$RUN_ID}
LOG_ROOT=${LOG_ROOT:-_rev/logs/run_all/$RUN_ID}
mkdir -p "$(dirname "$RESULT_ROOT")" "$(dirname "$LOG_ROOT")"
if ! mkdir "$RESULT_ROOT"; then
  echo "Refusing to reuse existing result root: $RESULT_ROOT" >&2
  exit 1
fi
if ! mkdir "$LOG_ROOT"; then
  echo "Refusing to reuse existing log root: $LOG_ROOT" >&2
  exit 1
fi
MANIFEST=$RESULT_ROOT/manifest.json
EXPECTED_ARTIFACTS=()

train_eval () {   # $1 config-name  $2 test-json  $3 log  $4 also_val(1/0)  $5 dataset
  local CFG=$1 TJ=$2 LG=$3 VALF=$4 DATASET=$5
  local EXP=${CFG}_s${SEED}_${RUN_ID}
  local CK=checkpoints/${EXP}/best.pth
  local MODEL
  case "$CFG" in
    *_traditional) MODEL=BaselineUNetPP ;;
    *_fcn) MODEL=FCNUNetPP ;;
    lift_*) MODEL=LiftUNetPP ;;
    *) MODEL=GRUUNetPP ;;
  esac
  EXPECTED_ARTIFACTS+=("$TJ|test|$DATASET|$SEED|segmentation|$MODEL|$EXP")
  echo "===== [$CFG seed=$SEED run=$RUN_ID] TRAIN $(date +%H:%M:%S) ====="
  if ! "$PYTHON_BIN" -m scripts.train --config "configs/experiments/${CFG}.yaml" --seed "$SEED" \
      --run_label "$RUN_ID" --refuse_existing_output > "$LG" 2>&1; then
    echo "[$CFG seed=$SEED] training failed" >&2
    tail -6 "$LG" >&2
    return 1
  fi
  if [ ! -f "$CK" ]; then
    echo "[$CFG seed=$SEED] training returned without $CK" >&2
    tail -6 "$LG" >&2
    return 1
  fi
  if [ "$VALF" = "1" ]; then
    local VJ="${TJ%_test.json}_val.json"
    EXPECTED_ARTIFACTS+=("$VJ|val|$DATASET|$SEED|segmentation|$MODEL|$EXP")
    rm -f "$VJ"
    "$PYTHON_BIN" -m scripts.evaluate --ckpt_path "$CK" --split val \
      --out_json "$VJ" >> "$LG" 2>&1
  fi
  rm -f "$TJ"
  "$PYTHON_BIN" -m scripts.evaluate --ckpt_path "$CK" --split test --out_json "$TJ" >> "$LG" 2>&1
  echo "[$CFG seed=$SEED] -> $TJ"
  tail -1 "$LG"
}

for SEED in "${SEED_ARRAY[@]}"; do
  # A: main clean comparison. Traditional runs first because recon uses their checkpoint.
  for CFG in rev_carvana_traditional rev_carvana_fcn rev_carvana_no_aux rev_carvana_fixed rev_carvana_tpls \
             rev_mnist_traditional   rev_mnist_fcn   rev_mnist_no_aux   rev_mnist_fixed   rev_mnist_tpls \
             rev_wbc_traditional     rev_wbc_fcn     rev_wbc_no_aux     rev_wbc_fixed     rev_wbc_tpls; do
    DATASET=${CFG#rev_}
    DATASET=${DATASET%%_*}
    train_eval "$CFG" "$RESULT_ROOT/${CFG}_s${SEED}_${RUN_ID}.json" \
      "$LOG_ROOT/${CFG}_s${SEED}_${RUN_ID}.log" 0 "$DATASET"
  done

  # C: WBC lifting ablation (validation and test).
  for L in gru srconv attn inr mamba kan; do
    CFG=lift_wbc_${L}
    train_eval "$CFG" "$RESULT_ROOT/${CFG}_s${SEED}_${RUN_ID}_test.json" \
      "$LOG_ROOT/${CFG}_s${SEED}_${RUN_ID}.log" 1 wbc
  done

  # B: clean reconstruction baselines (HSI=tradgi, CS=ADMM-L1).
  for ds in carvana mnist wbc; do
    EXP=rev_${ds}_traditional_s${SEED}_${RUN_ID}
    CK=checkpoints/${EXP}/best.pth
    CFGR=configs/experiments/rev_${ds}_traditional.yaml
    if [ ! -f "$CK" ]; then
      echo "recon $ds seed=$SEED: missing required checkpoint $CK" >&2
      exit 1
    fi
    for method in tradgi admm-l1; do
      RJ=$RESULT_ROOT/recon_${ds}_${method}_s${SEED}_${RUN_ID}.json
      EXPECTED_ARTIFACTS+=("$RJ|test|$ds|$SEED|recon:$method|BaselineUNetPP|$EXP")
      extra=()
      [ "$method" = "admm-l1" ] && extra=(--reg_weight 0.01 --rho 1.0 --steps 100)
      log=$LOG_ROOT/recon_${ds}_${method}_s${SEED}_${RUN_ID}.log
      echo "===== RECON $ds/$method seed=$SEED $(date +%H:%M:%S) ====="
      rm -f "$RJ"
      if ! "$PYTHON_BIN" -m scripts.reconstruct_eval --config "$CFGR" --ckpt_path "$CK" \
        --method "$method" --sampling_rate 0.03125 --split test --num_vis 6 \
        --save_vis_dir "$RESULT_ROOT/vis/recon_${method}_${ds}_s${SEED}" --out_json "$RJ" \
        "${extra[@]}" > "$log" 2>&1; then
        echo "recon $ds/$method seed=$SEED FAILED" >&2
        tail -5 "$log" >&2
        exit 1
      fi
      echo "recon $ds/$method seed=$SEED -> $RJ"
    done
  done
done

validation_args=()
for artifact in "${EXPECTED_ARTIFACTS[@]}"; do
  validation_args+=(--artifact "$artifact")
done
EXPECTED_COUNT=$((33 * ${#SEED_ARRAY[@]}))
"$PYTHON_BIN" -m scripts.validate_run_artifacts --mode "$MODE" --manifest "$MANIFEST" \
  --expected-count "$EXPECTED_COUNT" \
  "${validation_args[@]}"
echo "RUN_ALL_${MODE^^}_CLEAN_DONE $(date +%H:%M:%S)"
echo "For a complete paper audit, also run the separately documented TA, noise, rate, and mechanism sweeps."
