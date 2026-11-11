#!/usr/bin/env bash
# Result tables and figures.
# The first block reads the calculation tree (PPISQM_WORK) and writes data/;
# the second block needs only data/ and reproduces every statistic and figure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}

if [ "${SKIP_HARVEST:-0}" != 1 ]; then
  $PY scripts/harvest_scores.py            # protocol_scores.csv, protocol_metrics.csv
  $PY scripts/analyze_cpcmx.py             # cpcmx_metrics.csv
  $PY scripts/cpcmx_arms.py                # cpcmx_arms_*.csv
  $PY scripts/analyze_solvent_offsets.py   # solvent_offsets_*.csv
  $PY scripts/analyze_basin_weights.py     # basin_weights.csv/json
  $PY scripts/export_trajectory_scores.py  # scores_by_trajectory.csv
  $PY scripts/rmsd_analysis.py             # rmsd_analysis.csv
  $PY scripts/structure_statistics.py      # structure_statistics.csv
  $PY scripts/cpcmx_statistics.py          # cpcmx_statistics.csv
fi

$PY scripts/compare_methods.py             # method_comparison.csv
$PY scripts/ranking_bootstrap.py           # ranking_bootstrap_*.csv
$PY scripts/figures_main.py                # figures/figure1-8.png
$PY scripts/figures_si.py                  # figures/figureS1-S7.png
$PY scripts/figures_si_cpcmx.py            # figures/figureS8.png
$PY scripts/figure_toc.py --variant benchmark
