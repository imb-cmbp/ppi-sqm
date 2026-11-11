#!/usr/bin/env bash
# GFN2-xTB/CPCM-X on every scored model of all five sampling protocols.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
$PY -u scripts/score_cpcmx.py --sets iface aa bb contacts even10 \
  --workers "${CPCMX_WORKERS:-40}" --threads "${CPCMX_THREADS:-4}"
