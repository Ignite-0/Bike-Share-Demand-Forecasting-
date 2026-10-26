#!/usr/bin/env bash
# Upload the serve workflow's inputs (frozen models + 3 parquet files, ~330 MB) to the public HF dataset.
# Rerun after retraining or rebuilding station_hour (monthly trip refresh).
#   .venv/Scripts/hf auth login && bash deploy/hf_artifacts.sh
set -euo pipefail
REPO=${REPO:-himel05/citibike-artifacts}
HF=${HF:-.venv/Scripts/hf}
"$HF" repos create "$REPO" --type dataset --exist-ok
"$HF" upload "$REPO" . . --repo-type dataset --commit-message "inputs $(git rev-parse --short HEAD)" \
  --include "models/*.txt" --include "models/*.json" --include "data/processed/station_hour.parquet" \
  --include "data/processed/stations.parquet" --include "data/processed/station_labels.parquet"
echo "https://huggingface.co/datasets/$REPO"
