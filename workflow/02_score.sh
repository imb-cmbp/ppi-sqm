#!/usr/bin/env bash
# Nwat-MM/GBSA, frame extraction, clustering and PM6/PM7/xTB scoring for one or more
# trajectory sets.  Usage: workflow/02_score.sh r1 [r2 ...]
# Re-running is safe: finished cuby4/xtb jobs are read from cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORK=${PPISQM_WORK:-$PWD/work}
SCR="$WORK/screening_ff_md_sqm"

PM6_PRESETS=(
  pm6_d3h4_cosmo_nofc pm6_d3h4_cosmo_cuby pm6_d3h4_cosmo pm6_d3h4s_cosmo
  pm6_d3h4s_cosmo2 pm6_d3h4x_cosmo2 pm6_d3h4paren_cosmo pm6_d3_cosmo
  pm6_h4_cosmo_cuby pm6_dh2_cosmo pm6_dhplus_cosmo pm6_dhplus_cosmo2
  pm6_cosmo pm6_cosmo2 pm6_d3h4x pm7 pm7_cosmo2 pm6_ml pm6_ml_cosmo pm6_ml_cosmo2
)
XTB_VARIANTS=(gfn2_alpb gfn2_gbsa gfn2_cosmo gfn1_alpb gfn1_gbsa gfn1_cosmo)
PM6_WORKERS=${PM6_WORKERS:-72}
ML_WORKERS=${ML_WORKERS:-8}
XTB_WORKERS=${XTB_WORKERS:-72}
CLUSTER_WORKERS=${CLUSTER_WORKERS:-20}
MMGBSA_WORKERS=${MMGBSA_WORKERS:-16}
NWAT_WORKERS=${NWAT_WORKERS:-60}
EXTRACT_WORKERS=${EXTRACT_WORKERS:-24}

score_pm6() {  # score_pm6 <snapshot tree> <output dir> <trajectory>
  mkdir -p "$2"
  $PY scripts/score_mopac.py --in-base "$1" --out-base "$2" --ff ff14sb \
    --cutoffs 8 --presets "${PM6_PRESETS[@]}" --phases sp \
    --hydration wet --interface-mode dasa --dasa-cutoff 0.5 --polar-waters \
    --rep "$3" --max-waters 20 --n-snapshots 10 \
    --threads-per-sqm 1 --workers "$PM6_WORKERS" --ml-workers "$ML_WORKERS" \
    --min-free-gb 12 --out-csv "$2/detailed_$3_pm6series.csv"
}

score_xtb() {  # score_xtb <PM6 output dir holding the capped models> <output dir>
  mkdir -p "$2"
  $PY -u scripts/score_xtb.py --shell-base "$1" --out-base "$2" \
    --cutoff c8 --hydration wet --auto-frames --variants "${XTB_VARIANTS[@]}" \
    --workers "$XTB_WORKERS" --threads 1 --out-csv "$2/harvested_xtb.csv"
}

for rep in "$@"; do
  sysdir="$WORK/md/systems_propka_$rep"

  # Nwat-MM/GBSA on 100 frames of the final nanosecond
  $PY scripts/nwat_mmgbsa.py --nwat 0 10 20 30 40 --workers "$NWAT_WORKERS" --polar \
    --sysdir "$sysdir" --outdir "$WORK/md/mmgbsa_propka_${rep}_polar" \
    --results-csv "$WORK/md/mmgbsa_results_propka_${rep}_polar.csv"

  # Even-10: ten evenly spaced frames of the final nanosecond
  snaps="$WORK/screening_ff_md_sqm_propka_${rep}_w30"
  $PY scripts/extract_snapshots.py --sysdir "$sysdir" --rep "$rep" --out-base "$snaps" \
    --n-frames 10 --stride 100 --max-waters 30 --workers "$EXTRACT_WORKERS"
  score_pm6 "$snaps" "$SCR/_qmmm_trunc_propka/$rep" "$rep"
  score_xtb "$SCR/_qmmm_trunc_propka/$rep" "$SCR/_xtb_propka/$rep"

  # Four clustering schemes; centroids of basins holding more than 5% of the frames
  for scheme in "aa cluster_allatom.py --metric aa" "bb cluster_allatom.py --metric bb" \
                "iface_sc cluster_interface.py --mode iface_sc" \
                "contacts cluster_contacts.py --min-frac 0.05"; do
    read -r tag script extra <<< "$scheme"
    clusters="$WORK/screening_ff_md_sqm_propka_clust_$tag"
    basins="$WORK/screening_ff_md_sqm_propka_clust_${tag}_gt5"
    # shellcheck disable=SC2086
    $PY scripts/$script --sysdir "$sysdir" --reps "$rep" --out-base "$clusters" \
      --n-clusters 10 --workers "$CLUSTER_WORKERS" $extra
    if [ "$tag" = contacts ]; then
      basins="$clusters"
    else
      $PY scripts/extract_cluster_basins.py --cluster-base "$clusters" --out-base "$basins" \
        --reps "$rep" --min-frac 0.05 --sysdir "$sysdir" --workers "$EXTRACT_WORKERS"
    fi
    score_pm6 "$basins" "$SCR/_qmmm_trunc_propka_clust_${tag}_gt5/$rep" "$rep"
    score_xtb "$SCR/_qmmm_trunc_propka_clust_${tag}_gt5/$rep" "$SCR/_xtb_propka_clust_${tag}_gt5/$rep"
    $PY scripts/mmgbsa_centroids.py --snap-base "$basins" \
      --out-base "$WORK/md/mmgbsa_propka_clust_${tag}_gt5" --nwat 30 \
      --workers "$MMGBSA_WORKERS" --sysdir "$sysdir" --reps "$rep"
  done

  # MM/GB-minimised control on the all-atom basins (ACE/NME caps for tleap)
  out="$SCR/_qmmm_trunc_propka_clust_aa_gt5_mmsp_igb7/$rep"
  mkdir -p "$out"
  $PY -u scripts/score_mopac.py --in-base "$WORK/screening_ff_md_sqm_propka_clust_aa_gt5" \
    --out-base "$out" --ff ff14sb --cutoffs 8 --presets pm6_d3h4_cosmo_nofc --phases mm_sp \
    --hydration wet --interface-mode dasa --dasa-cutoff 0.5 --polar-waters \
    --rep "$rep" --max-waters 20 --n-snapshots 10 --caps ace_nme \
    --mm-igb 7 --mm-radii bondi --mm-maxcyc 1000 --mm-ncyc 300 --mm-workers 8 \
    --threads-per-sqm 1 --workers 20 --min-free-gb 12 --out-csv "$out/detailed_${rep}.csv"
done
