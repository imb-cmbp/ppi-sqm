#!/usr/bin/env python3
"""Cluster production frames on interface-residue RMSD only.

All-atom RMSD of the whole complex mixes distal loops that are stripped
before SQM scoring. This metric uses the same dASA interface residues as
the truncated shell.

Modes
  iface_sc      side-chain heavy atoms (default): packing / recognition
  iface_heavy   all heavy atoms of those residues (what the truncated
                solute actually contains, minus caps/waters)
  iface_polar   side-chain heavy + backbone N/O (H-bonding atoms)

Writes _cluster/summary.dat and cluster_info.json; basins with >= 5%
population are extracted by extract_cluster_basins.py.
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

from cluster_extract import traj_paths, PRODS  # noqa: E402
from cluster_representatives import REP_DIRS, _cpptraj, _frame_counts  # noqa: E402
import interface_dasa as ID  # noqa: E402
from common import PDB_IDS  # noqa: E402
from amber_setup import amber_env  # noqa: E402

OUT_BASE = REPO / "screening_ff_md_sqm_clust_iface_sc"
N_CLUSTERS = 10
STRIP_SOLVENT = ":WAT,HOH,T4D,Na+,Cl-"
MODES = ("iface_sc", "iface_heavy", "iface_polar")


def interface_mask(system, mode):
    resids = sorted(ID.get_mask(system, dasa_cutoff=0.5, polar=False) or [])
    if not resids:
        return None
    sel = ":" + ",".join(str(r) for r in resids)
    if mode == "iface_sc":
        return f"{sel}&!@H=&!@C,CA,N,O"
    if mode == "iface_polar":
        return f"{sel}&!@H=&!@CA,C"
    if mode == "iface_heavy":
        return f"{sel}&!@H="
    raise ValueError(mode)


def process(system, rep, out_base, ff_folder, mode, n_clusters, sysdir, force):
    env, _ = amber_env()
    _root, top, trajs = traj_paths(system, rep, sysdir)
    if not top.exists() or not all(t.exists() for t in trajs):
        return system, rep, "missing top/traj", None
    mask = interface_mask(system, mode)
    if not mask:
        return system, rep, "no dASA mask", None

    out_dir = Path(out_base) / system / ff_folder / rep
    work = out_dir / "_cluster"
    work.mkdir(parents=True, exist_ok=True)
    summary = work / "summary.dat"
    info_f = out_dir / "cluster_info.json"
    if not force and summary.exists() and info_f.exists():
        return system, rep, "cached", json.loads(info_f.read_text())

    counts = _frame_counts(trajs, top, work, env)
    if not counts:
        return system, rep, "frame count failed", None
    trajin = "".join(f"trajin {t}\n" for t in trajs)
    out, err, _code = _cpptraj(
        f"{trajin}"
        "autoimage\n"
        f"strip {STRIP_SOLVENT}\n"
        f"rms first {mask}\n"
        f"cluster hieragglo clusters {n_clusters} rms {mask} "
        "summary summary.dat info info.dat\n"
        "run\nquit\n",
        top, work, env, "clust")
    if not summary.exists():
        tail = (err or out or "")[-400:]
        return system, rep, f"clustering failed {tail}", None

    from cluster_extract import parse_all_clusters
    clusters = parse_all_clusters(summary)
    total = sum(counts)
    info = {
        "system": system, "rep": rep, "ff": ff_folder,
        "mode": mode, "rms_mask": mask, "n_clusters": n_clusters,
        "total_frames": total, "n_parsed": len(clusters),
        "dominant_frac": clusters[0]["frac"] if clusters else None,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, rep, "ok", info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--reps", nargs="*", default=sorted(REP_DIRS))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--ff-folder", default="ff14sb")
    ap.add_argument("--mode", choices=MODES, default="iface_sc")
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sysdir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(s, r) for s in args.systems for r in args.reps]
    print(f"interface clustering {len(jobs)} pairs mode={args.mode} "
          f"k={args.n_clusters} workers={args.workers}", flush=True)
    n_ok = 0
    pops = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, s, r, args.out_base, args.ff_folder,
                            args.mode, args.n_clusters, args.sysdir,
                            args.force): (s, r)
                for s, r in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            system, rep, status, info = fut.result()
            pop = info.get("dominant_frac") if info else None
            pop_s = f"{pop:.2f}" if pop is not None else "-"
            print(f"  [{k}/{len(jobs)}] {system} {rep}: {status} "
                  f"dom={pop_s}", flush=True)
            if status in ("ok", "cached"):
                n_ok += 1
                if pop is not None:
                    pops.append(pop)
    print(f"\n{n_ok}/{len(jobs)} ok")
    if pops:
        print(f"dominant basin: min {min(pops):.2f} "
              f"median {sorted(pops)[len(pops) // 2]:.2f} max {max(pops):.2f}")


if __name__ == "__main__":
    main()
