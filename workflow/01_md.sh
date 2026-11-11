#!/usr/bin/env bash
# System preparation and five independent MD trajectories per complex.
# Topologies go to $PPISQM_WORK/md/systems_propka; trajectory r<n> runs in
# $PPISQM_WORK/md/systems_propka_r<n>, starting from the same topology and coordinates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORK=${PPISQM_WORK:-$PWD/work}
GPUS=${GPUS:-0}

$PY scripts/build_topologies.py

for n in 1 2 3 4 5; do
  dest="$WORK/md/systems_propka_r$n"
  for src in "$WORK"/md/systems_propka/*/; do
    id=$(basename "$src")
    mkdir -p "$dest/$id"
    cp -n "$src/${id}_complex_wat.top" "$src/${id}_complex_wat.crd" "$dest/$id/"
  done
  $PY scripts/run_md.py --sysdir "$dest" --gpus $GPUS --log "$WORK/md/md_r$n.log"
done
