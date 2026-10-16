#!/usr/bin/env bash
set -euo pipefail

# Original 9/9 V4 expert chain with only the expert ablation changed to
# no_physics.  Data remain mounted remotely; default mode is preflight only.
BASE=${BASE:-/root/autodl-tmp}
CODE=${CODE:-$BASE/key_component_ablation_no_physics_mlp17_code_20260920}
DATASET_ROOT=${DATASET_ROOT:-$BASE/guizhou_v4/data/derived/windcloud_v4_all_1462_stationaware}
AUDIT_REPORT=${AUDIT_REPORT:-$DATASET_ROOT/audit.json}
CV_JSON=${CV_JSON:-$DATASET_ROOT/cv_17train_7eval.json}
SPLIT_CONFIG=${SPLIT_CONFIG:-$DATASET_ROOT/split_config.json}
AOD_REFERENCE=${AOD_REFERENCE:-$DATASET_ROOT/statistics/fold_0_aod_reference.json}
REFERENCE_IMPLEMENTATION=${REFERENCE_IMPLEMENTATION:-$CODE/reference/run_query_fusion_pilot_20260908.py}
RUN=${RUN:-$BASE/key_component_ablation_no_physics_mlp17_20260920}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
DEVICE=${DEVICE:-cuda}
RUN_TRAINING=${RUN_TRAINING:-0}
mkdir -p "$RUN/logs"
if [[ -e "$RUN/status.txt" ]]; then echo "Refusing to reuse RUN=$RUN" >&2; exit 2; fi
trap 'printf "failed\n" > "$RUN/status.txt"' ERR
printf 'preflight\n' > "$RUN/status.txt"

COMMON=(
  --dataset-root "$DATASET_ROOT" --audit-report "$AUDIT_REPORT" --cv "$CV_JSON"
  --split-config "$SPLIT_CONFIG" --aod-reference "$AOD_REFERENCE"
  --normalization-cache "$RUN/base_normalization.json" --fold 0 --seed 42
  --batch-size 4 --threads 4 --workers 0 --device "$DEVICE" --precision fp32
  --patience 10 --stages satellite,nwp,fusion --epochs satellite:80,nwp:80,fusion:50
  --ablation no_physics --satellite-path dual --routing-supervision none
  --routing-quality legacy
)

for required in "$CODE/src/ghi_hierarchical_v4/model.py" \
  "$CODE/scripts/train_hierarchical_799_v4.py" \
  "$CODE/scripts/cache_expert_states_17train_7eval.py" \
  "$CODE/scripts/train_mlp17_7eval_heads.py" \
  "$CODE/scripts/evaluate_mlp17_7eval_test.py" "$REFERENCE_IMPLEMENTATION" \
  "$DATASET_ROOT" "$AUDIT_REPORT" "$CV_JSON" "$SPLIT_CONFIG" "$AOD_REFERENCE"; do
  [[ -e "$required" ]] || { echo "Missing path: $required" >&2; exit 3; }
done

"$PYTHON" -u "$CODE/scripts/train_hierarchical_799_v4.py" "${COMMON[@]}" \
  --output-dir "$RUN/preflight" --preflight-only > "$RUN/logs/01_preflight.log" 2>&1
printf 'PREFLIGHT_PASS\n' > "$RUN/status.txt"
if [[ "$RUN_TRAINING" != 1 ]]; then
  echo "Preflight passed; set RUN_TRAINING=1 for formal training." | tee "$RUN/preflight_status.txt"
  exit 0
fi

printf 'training_expert\n' > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/train_hierarchical_799_v4.py" "${COMMON[@]}" \
  --output-dir "$RUN/no_physics_seed42" > "$RUN/logs/02_train_expert.log" 2>&1
printf 'caching_expert\n' > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/cache_expert_states_17train_7eval.py" \
  --dataset-root "$DATASET_ROOT" --split-config "$SPLIT_CONFIG" \
  --checkpoint "$RUN/no_physics_seed42/best_fusion.pt" \
  --normalization "$RUN/base_normalization.json" --aod-reference "$AOD_REFERENCE" \
  --output-dir "$RUN/cache" --device "$DEVICE" --ablation no_physics \
  --satellite-path dual --routing-quality legacy > "$RUN/logs/03_cache.log" 2>&1
printf 'training_external_mlp\n' > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/train_mlp17_7eval_heads.py" \
  --cache-dir "$RUN/cache" --reference-implementation "$REFERENCE_IMPLEMENTATION" \
  --output-dir "$RUN/dynamic_mlp" --device "$DEVICE" --seeds 42,43,44 \
  --epochs 40 --patience 8 > "$RUN/logs/04_train_mlp.log" 2>&1
printf 'closed_test\n' > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/evaluate_mlp17_7eval_test.py" \
  --dataset-root "$DATASET_ROOT" --split-config "$SPLIT_CONFIG" \
  --expert-checkpoint "$RUN/no_physics_seed42/best_fusion.pt" \
  --expert-normalization "$RUN/base_normalization.json" --expert-ablation no_physics \
  --satellite-path dual --routing-quality legacy --aod-reference "$AOD_REFERENCE" \
  --heads-dir "$RUN/dynamic_mlp" --reference-implementation "$REFERENCE_IMPLEMENTATION" \
  --output-dir "$RUN/test" --device "$DEVICE" > "$RUN/logs/05_closed_test.log" 2>&1
printf 'COMPLETE\n' > "$RUN/status.txt"
echo "Completed no_physics expert + external MLP chain: $RUN"
