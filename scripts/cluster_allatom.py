#!/usr/bin/env python3
"""Pick the centroid of the most populated cluster over all production frames.

Clusters every saved frame of prod1-prod4 (cpptraj hieragglo) on the RMSD of
all solute heavy atoms including side chains (--metric aa) or of backbone
CA/C/N (--metric bb); waters, ions and hydrogens are excluded.
Writes <out_base>/<SYS>/<ff>/<rep>/{dry,wet}_f1.pdb plus cluster_info.json.
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

from cluster_representatives import (  # noqa: E402
    REP_DIRS, _cpptraj, _frame_counts, _parse_summary, _global_to_file, rep_pad,
)
import interface_dasa as ID  # noqa: E402
from common import PDB_IDS, sanitize_wet_pdb_to_hoh  # noqa: E402
from amber_setup import amber_env  # noqa: E402

OUT_BASE = REPO / "screening_ff_md_sqm_clust_aa"
MAX_WATERS = 30
N_CLUSTERS = 10
# All four 1 ns production segments.
PRODS = ("prod1", "prod2", "prod3", "prod4")
# aa: all solute heavy atoms (no H, water or monovalent ions), side chains included.
# bb: backbone CA/C/N only.
RMS_MASKS = {
    "aa": "!(:WAT,HOH,T4D,Na+,Cl-)&!@H=",
    "bb": "!(:WAT,HOH,T4D,Na+,Cl-)&@CA,C,N",
}
RMS_MASK = RMS_MASKS["aa"]
STRIP_SOLVENT = ":WAT,HOH,T4D,Na+,Cl-"


def process(system, rep, out_base, ff_folder, force=False, n_clusters=N_CLUSTERS,
            sysdir=None, rms_mask=RMS_MASK, metric="aa"):
    env, _ = amber_env()
    sysdir = Path(sysdir) / system if sysdir else REP_DIRS[rep] / system
    top = (sysdir / f"{system}_complex_wat.top").resolve()
    trajs = [(sysdir / f"{system}_complex_wat.{p}.mdcrd").resolve() for p in PRODS]
    if not top.exists() or not all(t.exists() for t in trajs):
        missing = [t.name for t in trajs if not t.exists()]
        return system, rep, f"missing top/traj {missing}", None

    out_dir = Path(out_base).resolve() / system / ff_folder / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_dst, wet_dst = out_dir / "dry_f1.pdb", out_dir / "wet_f1.pdb"
    info_f = out_dir / "cluster_info.json"
    if not force and dry_dst.exists() and wet_dst.exists() and info_f.exists():
        return system, rep, "cached", json.loads(info_f.read_text())

    water_mask = ID.polar_mask_for_waters(system)
    if not water_mask:
        return system, rep, "no polar dASA mask", None

    work = out_dir / "_cluster"
    work.mkdir(exist_ok=True)

    counts = _frame_counts(trajs, top, work, env)
    if not counts:
        return system, rep, "frame count failed", None

    trajin = "".join(f"trajin {t}\n" for t in trajs)
    out, err, code = _cpptraj(
        f"{trajin}"
        "autoimage\n"
        f"strip {STRIP_SOLVENT}\n"
        f"rms first {rms_mask}\n"
        f"cluster hieragglo clusters {n_clusters} rms {rms_mask} "
        "summary summary.dat info info.dat\n"
        "run\nquit\n",
        top, work, env, "clust")
    summary = work / "summary.dat"
    if not summary.exists():
        tail = (err or out or "")[-400:]
        return system, rep, f"clustering failed {tail}", None
    best = _parse_summary(summary)
    if best is None:
        return system, rep, "no clusters parsed", None
    n_frames, centroid = best
    total = sum(counts)
    idx, local = _global_to_file(counts, centroid)
    if idx is None:
        return system, rep, f"centroid {centroid} out of range", None
    traj, fr = trajs[idx], local

    _cpptraj(
        f"trajin {traj} {fr} {fr}\n"
        "autoimage\n"
        f"strip {STRIP_SOLVENT}\n"
        f"trajout {dry_dst.name}\nrun\nquit\n",
        top, work, env, "dry")
    src = work / dry_dst.name
    if src.exists():
        src.replace(dry_dst)

    for f in work.glob("clo.*"):
        f.unlink(missing_ok=True)
    _cpptraj(
        f"trajin {traj} {fr} {fr}\n"
        "center @CA,C,N mass origin\n"
        "image origin center\n"
        "strip :Na+,Cl-\n"
        f"closest {MAX_WATERS} {water_mask} noimage outprefix clo\n"
        f"trajout {wet_dst.name} nobox\nrun\nquit\n",
        top, work, env, "wet")
    gen = list(work.glob("clo.*"))
    if gen:
        _cpptraj(f"trajin {gen[0].name} 1 1\n"
                 f"trajout {wet_dst.name}\nrun\nquit\n",
                 top, work, env, "wet2")
    w = work / wet_dst.name
    if w.exists():
        w.replace(wet_dst)
    if wet_dst.exists():
        sanitize_wet_pdb_to_hoh(wet_dst)

    if not (dry_dst.exists() and wet_dst.exists()):
        return system, rep, "extraction failed", None

    info = {
        "system": system, "rep": rep, "ff": ff_folder,
        "prods": list(PRODS), "total_frames": total,
        "cluster_frames": n_frames,
        "cluster_population": round(n_frames / total, 4),
        "centroid_global_frame": centroid,
        "source_traj": traj.name, "source_local_frame": fr,
        "n_clusters": n_clusters, "rms_mask": rms_mask,
        "metric": ("backbone_CA_C_N" if metric == "bb"
                   else "all_solute_heavy_atoms_incl_sidechains"),
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, rep, "ok", info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--reps", nargs="*", default=sorted(REP_DIRS))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--ff-folder", default="ff14sb",
                    help="Subfolder name under <SYS>/ (ff14sb or a99sb-disp).")
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sysdir", default=None,
                    help="Override trajectory root (use for a99sb-disp: "
                         "md/systems_propka_r1). Default is the "
                         "md/systems layout.")
    ap.add_argument("--metric", choices=sorted(RMS_MASKS), default="aa",
                    help="aa = all solute heavy atoms; bb = backbone CA/C/N.")
    args = ap.parse_args()
    rms_mask = RMS_MASKS[args.metric]

    jobs = [(s, r) for s in args.systems for r in args.reps]
    print(f"{args.metric} clustering {len(jobs)} pairs on {PRODS} "
          f"mask={rms_mask} k={args.n_clusters} workers={args.workers}",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, s, r, args.out_base, args.ff_folder,
                            args.force, args.n_clusters, args.sysdir,
                            rms_mask, args.metric): (s, r)
                for s, r in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            system, rep, status, info = fut.result()
            pop = f"{info['cluster_population']:.2f}" if info else "-"
            frame = info["centroid_global_frame"] if info else "-"
            print(f"  [{k}/{len(jobs)}] {system} {rep}: {status} "
                  f"pop={pop} frame={frame}", flush=True)
            rows.append((system, rep, status, info))

    ok = [r for r in rows if r[2] in ("ok", "cached")]
    print(f"\n{len(ok)}/{len(rows)} representatives available")
    bad = [(s, r, st) for s, r, st, _ in rows if st not in ("ok", "cached")]
    if bad:
        print("failures:")
        for s, r, st in bad:
            print(f"  {s} {rep_pad(r)} {st}")
    pops = [i["cluster_population"] for _s, _r, st, i in rows
            if i and st in ("ok", "cached")]
    if pops:
        print(f"dominant-cluster population: min {min(pops):.2f} "
              f"median {sorted(pops)[len(pops) // 2]:.2f} max {max(pops):.2f}")


if __name__ == "__main__":
    main()
