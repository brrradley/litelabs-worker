#!/usr/bin/env bash
set -euo pipefail

export LITELABS_BENCHMARK_ZIP_URL="https://literecords.com/tmp/disturbia_test.zip"
export LITELABS_BENCHMARK_OUTPUT="/workspace/litelabs-research/disturbia_vocal_final_validation_v8.json"
export LITELABS_BENCHMARK_MODEL_TIMEOUT="1800"

mkdir -p /workspace/litelabs-research

echo "[LiteLABS final vocal] forcing known-good Disturbia benchmark source"
echo "[LiteLABS final vocal] output: ${LITELABS_BENCHMARK_OUTPUT}"

python -u /app/experimental_vocal_cross_song_validation.py

echo "[LiteLABS final vocal] benchmark complete; serving /workspace/litelabs-research on port 8888"
exec python -m http.server 8888 --bind 0.0.0.0 --directory /workspace/litelabs-research
