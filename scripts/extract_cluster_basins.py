#!/usr/bin/env python3
"""Extract every RMSD-cluster centroid whose basin is above a population cutoff.

Reads cpptraj cluster summaries under --cluster-base (default: output of
cluster_allatom.py) and writes wet/dry_f1..fK plus basins.json with the basin
populations used for population-weighted scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WORK  # noqa: E402

REPO = WORK

from cluster_extract import parse_all_clusters, extract_frame  # noqa: E402
from cluster_representatives import REP_DIRS  # noqa: E402
from common import PDB_IDS  # noqa: E402

CLUSTER_BASE = REPO / "screening_ff_md_sqm_clust_aa"
OUT_BASE = REPO / "screening_ff_md_sqm_clust_aa_gt5"
MIN_FRAC = 0.05


def process(system, rep, cluster_base, out_base, ff_folder, min_frac, force,
            sysdir=None):
    summary = Path(cluster_base) / system / ff_folder / rep / "_cluster" / "summary.dat"
    if not summary.exists():
        return system, rep, "no summary", None
    kept = [c for c in parse_all_clusters(summary) if c["frac"] >= min_frac]
    if not kept:
        return system, rep, "no basin above cutoff", None
    out_dir = Path(out_base) / system / ff_folder / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    info_f = out_dir / "basins.json"
    if not force and info_f.exists() and all(
            (out_dir / f"wet_f{i}.pdb").exists() for i in range(1, len(kept) + 1)):
        return system, rep, "cached", json.loads(info_f.read_text())

    basins = []
    for i, c in enumerate(kept, 1):
        dry = out_dir / f"dry_f{i}.pdb"
        wet = out_dir / f"wet_f{i}.pdb"
        ok, extra = extract_frame(system, rep, c["centroid"], dry, wet, sysdir)
        if not ok:
            return system, rep, f"extract f{i} failed: {extra}", None
        basins.append({
            "snap": i,
            "cluster": c["cluster"],
            "n_frames": c["n_frames"],
            "frac": c["frac"],
            "centroid_global_frame": c["centroid"],
            **({} if not isinstance(extra, dict) else extra),
        })
    info = {
        "system": system, "rep": rep, "ff": ff_folder,
        "min_frac": min_frac, "n_basins": len(basins),
        "covered_frac": round(sum(b["frac"] for b in basins), 4),
        "basins": basins,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, rep, "ok", info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--reps", nargs="*", default=sorted(REP_DIRS))
    ap.add_argument("--cluster-base", default=str(CLUSTER_BASE))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--ff-folder", default="ff14sb")
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sysdir", default=None,
                    help="Trajectory root for extract_frame (PROPKA replica dirs).")
    args = ap.parse_args()

    jobs = [(s, r) for s in args.systems for r in args.reps]
    print(f"extracting basins frac>={args.min_frac} for {len(jobs)} pairs",
          flush=True)
    n_ok, n_basins = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, s, r, args.cluster_base, args.out_base,
                            args.ff_folder, args.min_frac, args.force,
                            args.sysdir): (s, r)
                for s, r in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            system, rep, status, info = fut.result()
            nb = info["n_basins"] if info else 0
            cov = info["covered_frac"] if info else 0
            print(f"  [{k}/{len(jobs)}] {system} {rep}: {status} "
                  f"n={nb} covered={cov:.2f}", flush=True)
            if status in ("ok", "cached") and info:
                n_ok += 1
                n_basins += info["n_basins"]
    print(f"\n{n_ok}/{len(jobs)} ok, {n_basins} centroids extracted "
          f"(mean {n_basins / max(n_ok, 1):.1f} per replica)")


if __name__ == "__main__":
    main()
