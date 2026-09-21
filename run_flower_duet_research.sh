#!/usr/bin/env bash
set -euo pipefail

TRACK_URL="https://literecords.com/tmp/02%20Lakme_%20Sous%20le%20dome%20Epais%20%28The%20Flower%20Duet%29.mp3"
TRACK_NAME="02 Lakme_ Sous le dome Epais (The Flower Duet).mp3"
OUTPUT_DIR="/workspace/litelabs-results"

mkdir -p "${OUTPUT_DIR}"

echo "[LiteLABS research] Flower Duet combined research test"
echo "[LiteLABS research] dual-lead + instrument inventory + G400"
echo "[LiteLABS research] output directory: ${OUTPUT_DIR}"

python -u /app/research_run_experimental.py \
  "${TRACK_URL}" \
  --filename "${TRACK_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --timeout 3600

echo
echo "[LiteLABS research] COMPLETE"
echo "[LiteLABS research] Result pack is ready in ${OUTPUT_DIR}"
echo "[LiteLABS research] Starting download server on port 8888"
echo "[LiteLABS research] Open the Pod HTTP service for port 8888 to download the ZIP."

exec python -m http.server 8888 --bind 0.0.0.0 --directory "${OUTPUT_DIR}"
