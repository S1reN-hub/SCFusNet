#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp}
CODE=${CODE:-$BASE/guizhou_mlp17_7eval_code_20260909}
DATA_PARENT=${DATA_PARENT:-$BASE/guizhou_v4/data/derived}
DEVELOPMENT=${DEVELOPMENT:-$DATA_PARENT/windcloud_v4_development_1241}
CONFIRMATION=${CONFIRMATION:-$DATA_PARENT/windcloud_v4_confirmation_221}
COMBINED=${COMBINED:-$DATA_PARENT/windcloud_v4_all_1462_stationaware}
RUN=${RUN:-$BASE/guizhou_mlp17_7eval_run_20260909}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}

mkdir -p "$RUN/logs"
trap 'printf "failed\n" > "$RUN/status.txt"' ERR
printf "preparing\n" > "$RUN/status.txt"

"$PYTHON" -u "$CODE/scripts/prepare_combined_1462_dataset.py" \
  --development-root "$DEVELOPMENT" \
  --confirmation-root "$CONFIRMATION" \
  --split-config "$CODE/configs/dataset_split_17train_7eval_expanded_20260909.json" \
  --output-root "$COMBINED" \
  > "$RUN/logs/01_prepare_dataset.log" 2>&1

"$PYTHON" -u "$CODE/scripts/audit_windcloud_v4_dataset.py" \
  --dataset-root "$COMBINED" \
  --output "$COMBINED/audit.json" \
  --inspect-labels \
  > "$RUN/logs/02_dataset_audit.log" 2>&1

printf "preflight\n" > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/train_hierarchical_799_v4.py" \
  --dataset-root "$COMBINED" \
  --audit-report "$COMBINED/audit.json" \
  --cv "$COMBINED/cv_17train_7eval.json" \
  --split-config "$COMBINED/split_config.json" \
  --aod-reference "$COMBINED/statistics/fold_0_aod_reference.json" \
  --normalization-cache "$RUN/base_normalization.json" \
  --output-dir "$RUN/preflight" \
  --fold 0 --seed 42 --batch-size 4 --threads 4 --workers 0 \
  --device cuda --precision fp32 --patience 10 \
  --stages satellite,nwp,fusion \
  --epochs satellite:80,nwp:80,fusion:50 \
  --ablation fixed_router --satellite-path dual \
  --routing-supervision none --routing-quality legacy \
  --preflight-only \
  > "$RUN/logs/03_preflight.log" 2>&1

printf "training_experts\n" > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/train_hierarchical_799_v4.py" \
  --dataset-root "$COMBINED" \
  --audit-report "$COMBINED/audit.json" \
  --cv "$COMBINED/cv_17train_7eval.json" \
  --split-config "$COMBINED/split_config.json" \
  --aod-reference "$COMBINED/statistics/fold_0_aod_reference.json" \
  --normalization-cache "$RUN/base_normalization.json" \
  --output-dir "$RUN/base_seed42" \
  --fold 0 --seed 42 --batch-size 4 --threads 4 --workers 0 \
  --device cuda --precision fp32 --patience 10 \
  --stages satellite,nwp,fusion \
  --epochs satellite:80,nwp:80,fusion:50 \
  --ablation fixed_router --satellite-path dual \
  --routing-supervision none --routing-quality legacy \
  > "$RUN/logs/04_train_experts.log" 2>&1

printf "caching_experts\n" > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/cache_expert_states_17train_7eval.py" \
  --dataset-root "$COMBINED" \
  --split-config "$COMBINED/split_config.json" \
  --checkpoint "$RUN/base_seed42/best_fusion.pt" \
  --normalization "$RUN/base_normalization.json" \
  --aod-reference "$COMBINED/statistics/fold_0_aod_reference.json" \
  --output-dir "$RUN/cache" --device cuda \
  > "$RUN/logs/05_cache_experts.log" 2>&1

printf "training_dynamic_mlp\n" > "$RUN/status.txt"
"$PYTHON" -u "$CODE/scripts/train_mlp17_7eval_heads.py" \
  --cache-dir "$RUN/cache" \
  --reference-implementation "$CODE/reference/run_query_fusion_pilot_20260908.py" \
  --output-dir "$RUN/dynamic_mlp" \
  --device cuda --seeds 42,43,44 --epochs 40 --patience 8 \
  > "$RUN/logs/06_train_dynamic_mlp.log" 2>&1

printf "validation_complete_test_unread\n" > "$RUN/status.txt"
printf "TRAINING_AND_VALIDATION_COMPLETE\n"
