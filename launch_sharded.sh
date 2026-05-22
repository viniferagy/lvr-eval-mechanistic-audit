#!/usr/bin/env bash
# =============================================================================
# launch_sharded.sh — 把工作分摊到 4×4090
#
# 策略：按 (模型 × 阶段) 切 job，通过 run_and_hold.sh 在外层锁定 4 张 GPU。
# 4 张卡的典型分配（Qwen2.5-VL-7B/LVR-7B 两模型 × BF-1/CF-2 两阶段 = 4 个 job）:
#   GPU0: qwen2_5_vl_7b + BF-1
#   GPU1: qwen2_5_vl_7b + CF-2
#   GPU2: lvr_7b + BF-1   (若 LVR-7B 未下载完成，会在 run_all.py 内等待)
#   GPU3: lvr_7b + CF-2   (若 LVR-7B 未下载完成，会在 run_all.py 内等待)
# 各 job 把结果写进各自 run dir，最后合并跑一次 analysis。
#
# 用法: bash launch_sharded.sh
# =============================================================================
set -e
CONFIG=${1:-config.yaml}
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
  elif [ -x "../.venv/bin/python" ]; then
    PYTHON_BIN="../.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="./runs/sharded_${STAMP}"
mkdir -p "$ROOT"
JOB_PIDS=()
JOB_LABELS=()

run_job () {  # $1=gpu  $2=model  $3=phase
  local gpu=$1 model=$2 phase=$3
  CUDA_VISIBLE_DEVICES=$1 \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  MPLCONFIGDIR=/tmp/matplotlib-lvr-eval \
  "$PYTHON_BIN" run_all.py --config "$CONFIG" --models "$2" --only "$3" \
    --device cuda:0 --output-root "$ROOT" --run-name "${model}_${phase}" --no-analysis \
    > "${ROOT}/log_${2}_${3}.txt" 2>&1 &
  local pid=$!
  JOB_PIDS+=("$pid")
  JOB_LABELS+=("GPU${gpu}:${model}:${phase}")
  echo "  launched: GPU$1  $2  $3  (pid $pid)"
}

wait_jobs () {
  local status=0
  for idx in "${!JOB_PIDS[@]}"; do
    local pid=${JOB_PIDS[$idx]}
    local label=${JOB_LABELS[$idx]}
    if wait "$pid"; then
      echo "  finished: $label"
    else
      local rc=$?
      echo "  FAILED:   $label (exit $rc)" >&2
      status=1
    fi
  done
  return "$status"
}

echo "[sharded] root=$ROOT"
echo "[sharded] python=$PYTHON_BIN"
echo "[sharded] preflight..."
"$PYTHON_BIN" - "$CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

from pipeline.data import load_probe_set

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

lvr_cfg = (cfg.get("models") or {}).get("lvr_7b") or {}
lvr_source = lvr_cfg.get("lvr_source_path") or os.environ.get("LVR_SOURCE_PATH") or "../lvr"
lvr_source_path = Path(os.path.expandvars(os.path.expanduser(str(lvr_source))))
if not lvr_source_path.is_absolute():
    lvr_source_path = (Path.cwd() / lvr_source_path).resolve()
lvr_marker = lvr_source_path / "src" / "model" / "qwen_lvr_model.py"
if not lvr_marker.is_file():
    raise SystemExit(
        f"LVR source missing: {lvr_marker}. "
        "Set models.lvr_7b.lvr_source_path or LVR_SOURCE_PATH."
    )

samples = load_probe_set(cfg["data"])
if not samples:
    raise SystemExit(
        "probe set is empty. For LVR JSON, check data.json_path, image_root, "
        "require_lvr_placeholder, and whether Visual-CoT images have been extracted."
    )
print(f"[sharded] preflight ok: samples={len(samples)} lvr_source={lvr_source_path}")
PY
# 注意：每个进程内部用 cuda:0，因为 CUDA_VISIBLE_DEVICES 已把目标卡映射成 0
run_job 0 qwen2_5_vl_7b bf1
run_job 1 qwen2_5_vl_7b cf2
run_job 2 lvr_7b bf1
run_job 3 lvr_7b cf2
wait_jobs
echo "[sharded] all jobs finished. 各 job 的 run dir 在 ./runs/ 下（时间戳区分）。"
echo "[sharded] 如需统一出图，把各 job 产出的 bf1_*.json / cf2_*.json 收集到同一目录，再运行:"
echo "          python merge_and_analyze.py --dir <收集目录>"
