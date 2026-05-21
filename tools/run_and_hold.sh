#!/usr/bin/env bash
# Usage: run_and_hold.sh <gpu_ids> <command...>
# 例: bash ../../run_and_hold.sh 0,1,2,3 launch_sharded.sh
set -u

if [ $# -lt 2 ]; then
  echo "Usage: $0 <gpu_ids> <command...>"
  exit 1
fi

GPU_IDS="$1"; shift

# 用绝对路径锁定 hold_gpu.py —— 与调用时的 cwd 无关
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLD_SCRIPT="${HOLD_SCRIPT:-$SCRIPT_DIR/hold_gpu.py}"
if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in \
    "$SCRIPT_DIR/../.venv/bin/python" \
    "$SCRIPT_DIR/../../05200/.venv/bin/python" \
    "$SCRIPT_DIR/05200/.venv/bin/python"; do
    if [ -x "$candidate" ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [ ! -f "$HOLD_SCRIPT" ]; then
  echo "[run_and_hold] hold script not found: $HOLD_SCRIPT" >&2
  exit 1
fi

SAFE_IDS="${GPU_IDS//[^0-9A-Za-z]/_}"
PID_FILE="/tmp/gpu_hold_${USER}_${SAFE_IDS}.pid"
LOG_FILE="/tmp/gpu_hold_${USER}_${SAFE_IDS}.log"

kill_previous() {
  if [ -f "$PID_FILE" ]; then
    local old; old=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
      echo "[run_and_hold] killing previous holder (PID=$old) on GPU $GPU_IDS"
      kill -TERM "-$old" 2>/dev/null || kill -TERM "$old" 2>/dev/null || true
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$old" 2>/dev/null || break
        sleep 0.3
      done
      kill -KILL "-$old" 2>/dev/null || kill -KILL "$old" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  # pid file can go stale if the holder was started by an older wrapper or the
  # shell lost track of the process group. Clean up our own stale holders too.
  local stale_pids
  stale_pids=$(pgrep -u "$USER" -f "$HOLD_SCRIPT" 2>/dev/null || true)
  if [ -n "${stale_pids:-}" ]; then
    echo "[run_and_hold] killing stale holder process(es): ${stale_pids//$'\n'/ }"
    while read -r stale; do
      [ -n "$stale" ] || continue
      kill -TERM "-$stale" 2>/dev/null || kill -TERM "$stale" 2>/dev/null || true
    done <<< "$stale_pids"
    sleep 1
    while read -r stale; do
      [ -n "$stale" ] || continue
      kill -KILL "-$stale" 2>/dev/null || kill -KILL "$stale" 2>/dev/null || true
    done <<< "$stale_pids"
  fi
}

start_holder() {
  echo "[run_and_hold] starting holder on GPU $GPU_IDS ..."
  CUDA_VISIBLE_DEVICES="$GPU_IDS" setsid nohup "$PYTHON_BIN" "$HOLD_SCRIPT" \
      >"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" >"$PID_FILE"
  echo "[run_and_hold] holder PID=$pid (pgid=$pid), log=$LOG_FILE"
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[run_and_hold] holder exited immediately; recent log:" >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    return 1
  fi
}

# 关键改动：裸文件名 (不含 / )，PATH 里找不到，但 cwd 下存在 -> 用 ./文件名
# 这样 `bash ../../run_and_hold.sh 0,1 launch_sharded.sh` 才能在子目录里跑起来
if [[ "$1" != */* ]] && ! command -v "$1" >/dev/null 2>&1 && [ -f "./$1" ]; then
  if [ -x "./$1" ]; then
    set -- "./$1" "${@:2}"
  else
    set -- bash "./$1" "${@:2}"   # 没有 x 权限就用 bash 启
  fi
fi

kill_previous

trap 'echo "[run_and_hold] interrupted"; trap - INT TERM; start_holder; exit 130' INT TERM

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "[run_and_hold] cwd=$(pwd)"
echo "[run_and_hold] CUDA_VISIBLE_DEVICES=$GPU_IDS  running: $*"
"$@"                              # 在调用时的 cwd 下执行，相对路径全部生效
status=$?
echo "[run_and_hold] command exited with status $status"

trap - INT TERM
start_holder

exit "$status"
