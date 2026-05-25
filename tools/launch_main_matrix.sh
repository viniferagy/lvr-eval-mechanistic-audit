#!/usr/bin/env bash
# Launch task/model/metric shards for the W16 main matrix.
set -euo pipefail

CONFIGS=()
MODELS="qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b"
METRICS="all"
GPUS="0,1,2,3"
RUN_ROOT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/launch_main_matrix.sh \
    --configs config.main_spd_n1000.yaml,config.main_maze_n1000.yaml \
    --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
    --metrics all \
    --gpus 0,1,2,3 \
    --run-root runs/w16_main_matrix_t1_t2_t3

Run through:
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh ...
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --configs)
      IFS=',' read -r -a CONFIGS <<< "$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "--configs is required" >&2
  exit 2
fi
if [ -z "$RUN_ROOT" ]; then
  RUN_ROOT="./runs/w16_main_matrix_$(date +%Y%m%d_%H%M%S)"
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
  elif [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
IFS=',' read -r -a METRIC_GROUPS <<< "$METRICS"

mkdir -p "$RUN_ROOT"
JOB_PIDS=()
JOB_LABELS=()
NEXT_GPU=0

safe_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

metrics_for_config() {
  local config=$1
  local metrics=$2
  if [ "$metrics" != "all" ]; then
    echo "$metrics"
    return
  fi
  "$PYTHON_BIN" - "$config" <<'PY'
import sys
import yaml
from run_all import selected_metric_ids

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(",".join(selected_metric_ids(["all"], cfg)))
PY
}

launch_job() {
  local gpu=$1 config=$2 model=$3 metric_csv=$4
  local config_base model_safe metric_safe run_name log_path
  config_base=$(basename "$config" .yaml)
  model_safe=$(safe_name "$model")
  metric_safe=$(safe_name "$metric_csv")
  run_name="${config_base}_${model_safe}_${metric_safe}"
  log_path="${RUN_ROOT}/log_${run_name}.txt"
  IFS=',' read -r -a metric_args <<< "$metric_csv"
  local cmd=( "$PYTHON_BIN" run_all.py --config "$config" --models "$model" --only "${metric_args[@]}" --device cuda:0 --output-root "$RUN_ROOT" --run-name "$run_name" --no-analysis )
  echo "[matrix] GPU${gpu} ${config} ${model} ${metric_csv}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lvr-eval}" \
  "${cmd[@]}" > "$log_path" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_LABELS+=("GPU${gpu}:${run_name}")
}

wait_some_if_needed() {
  while [ "${#JOB_PIDS[@]}" -ge "${#GPU_LIST[@]}" ]; do
    local pid=${JOB_PIDS[0]}
    local label=${JOB_LABELS[0]}
    if wait "$pid"; then
      echo "[matrix] finished $label"
    else
      local rc=$?
      echo "[matrix] FAILED $label exit=$rc" >&2
      exit "$rc"
    fi
    JOB_PIDS=("${JOB_PIDS[@]:1}")
    JOB_LABELS=("${JOB_LABELS[@]:1}")
  done
}

for config in "${CONFIGS[@]}"; do
  if [ ! -f "$config" ]; then
    echo "missing config: $config" >&2
    exit 2
  fi
  resolved_metrics=$(metrics_for_config "$config" "$METRICS")
  IFS=';' read -r -a metric_cells <<< "${resolved_metrics//|/;}"
  if [ "${#metric_cells[@]}" -eq 0 ]; then
    metric_cells=("$resolved_metrics")
  fi
  for model in "${MODEL_LIST[@]}"; do
    for metric_csv in "${metric_cells[@]}"; do
      [ -n "$metric_csv" ] || continue
      wait_some_if_needed
      gpu="${GPU_LIST[$NEXT_GPU]}"
      NEXT_GPU=$(( (NEXT_GPU + 1) % ${#GPU_LIST[@]} ))
      launch_job "$gpu" "$config" "$model" "$metric_csv"
    done
  done
done

for idx in "${!JOB_PIDS[@]}"; do
  pid=${JOB_PIDS[$idx]}
  label=${JOB_LABELS[$idx]}
  if wait "$pid"; then
    echo "[matrix] finished $label"
  else
    rc=$?
    echo "[matrix] FAILED $label exit=$rc" >&2
    exit "$rc"
  fi
done

echo "[matrix] all jobs finished -> $RUN_ROOT"
echo "[matrix] merge with: $PYTHON_BIN tools/merge_main_matrix.py $RUN_ROOT"
