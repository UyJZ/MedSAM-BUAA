#!/usr/bin/env bash
# Run the full MedSAM experiment sweep in parallel (up to 4 concurrent jobs).
# Extra CLI flags are forwarded to run_medsam_experiments.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/media/space/zjy/anaconda3/envs/zjy/bin/python"

# ===================== config =====================
DEFAULT_DATASETS=(kvasir glas busi isic brats)

DEFAULT_MODELS=(
  sam
  sam_adapter
  sam_lora
  sam_prompt
  sam_adapter_lora
  sam_adapter_prompt
  sam_lora_prompt
  sam_adapter_lora_prompt
)

DEFAULT_RATIOS=(0.1 0.5 0.7)
DEFAULT_DEVICES=(cuda:0)

MAX_JOBS=${#DEFAULT_DEVICES[@]}   # 并行数 = GPU 数
# ==================================================

run_dataset() {
  local dataset="$1"
  local device="$2"
  shift 2

  echo "[run_all_medsam_tests] START dataset=${dataset} device=${device}"

  CUDA_VISIBLE_DEVICES="${device##*:}" \
  "$PYTHON_BIN" run_medsam_experiments.py \
    --datasets "$dataset" \
    --models "${DEFAULT_MODELS[@]}" \
    --train-ratios "${DEFAULT_RATIOS[@]}" \
    --epochs 25 \
    --batch-size 2 \
    --num-workers 4 \
    --device "$device" \
    --task-prefix "full_sweep_${dataset}" \
    "$@"

  echo "[run_all_medsam_tests] DONE  dataset=${dataset} device=${device}"
}

job_count=0
task_idx=0

for dataset in "${DEFAULT_DATASETS[@]}"; do
  device="${DEFAULT_DEVICES[$((task_idx % ${#DEFAULT_DEVICES[@]}))]}"

  run_dataset "$dataset" "$device" "$@" &

  job_count=$((job_count + 1))
  task_idx=$((task_idx + 1))

  # 若并发达到上限，等待任意一个任务结束
  if (( job_count >= MAX_JOBS )); then
    wait -n
    job_count=$((job_count - 1))
  fi
done

# 等待所有后台任务完成
wait

echo "[run_all_medsam_tests] ALL DATASETS FINISHED"
echo "[run_all_medsam_tests] datasets : ${DEFAULT_DATASETS[*]}"
echo "[run_all_medsam_tests] models   : ${DEFAULT_MODELS[*]}"
echo "[run_all_medsam_tests] ratios   : ${DEFAULT_RATIOS[*]}"
