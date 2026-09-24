#!/usr/bin/env bash
set -euo pipefail

MSST_DIR="${MSST_DIR:-/opt/music-source-separation-training}"
DATA_ROOT="${DATA_ROOT:-/workspace/set-drumsep-data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/set-drumsep-results}"
CONFIG="${CONFIG:-/app/training/drumsep_melband_4stem.yaml}"
START_CKPT="${START_CKPT:-}"

mkdir -p "${RESULTS_ROOT}"

ARGS=(
  python -u "${MSST_DIR}/train.py"
  --model_type mel_band_roformer
  --config_path "${CONFIG}"
  --results_path "${RESULTS_ROOT}"
  --data_path "${DATA_ROOT}/train"
  --valid_path "${DATA_ROOT}/valid"
  --dataset_type 4
  --num_workers "${NUM_WORKERS:-4}"
  --device_ids 0
  --metrics sdr si_sdr bleedless fullness
  --metric_for_scheduler sdr
  --pre_valid
  --each_metrics_in_name
  --load_only_compatible_weights
)

if [[ -n "${START_CKPT}" ]]; then
  ARGS+=(--start_check_point "${START_CKPT}")
fi

echo "[SET DrumSep training] MSST: ${MSST_DIR}"
echo "[SET DrumSep training] data: ${DATA_ROOT}"
echo "[SET DrumSep training] results: ${RESULTS_ROOT}"
echo "[SET DrumSep training] start checkpoint: ${START_CKPT:-from scratch}"
printf ' %q' "${ARGS[@]}"
echo

exec "${ARGS[@]}"
