#!/usr/bin/env python3
"""Pick one representative snapshot per (system, replica) by interface RMSD clustering.

Clusters prod2-prod4 on interface heavy-atom RMSD (cpptraj hieragglo) and
extracts the centroid of the most populated cluster (cf. Yurenko et al.,
JCIM 2025, 65, 5690, on snapshot selection for SQM scoring). Writes
<out_base>/<SYS>/ff14sb/<rep>/{dry,wet}_f1.pdb in the layout read by
score_mopac.py, plus cluster_info.json with the cluster population and frame.
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

from amber_setup import amber_bin, amber_env  # noqa: E402
import interface_dasa as ID  # noqa: E402
from common import safe_run  # noqa: E402
from common import PDB_IDS, sanitize_wet_pdb_to_hoh  # noqa: E402

AR = REPO / "md"
# rep1 lives in the unsuffixed directory; the rest are suffixed.
REP_DIRS = {"rep1": AR / "systems", "rep2": AR / "systems_rep2",
            "rep3": AR / "systems_rep3", "rep4": AR / "systems_rep4",
            "rep5": AR / "systems_rep5"}
OUT_BASE = REPO / "screening_ff_md_sqm_clust"
MAX_WATERS = 30       # as in extract_snapshots.py; scoring may keep fewer
N_CLUSTERS = 5
# prod1 is skipped because the interface is still relaxing after
# equilibration; the last 3 ns are clustered.
PRODS = ("prod2", "prod3", "prod4")


def _cpptraj(inp, top, wd, env, tag):
    (wd / f"{tag}.cpptraj").write_text(inp)
    return safe_run(f"{amber_bin('cpptraj')} -i {tag}.cpptraj -p {top}", wd, env)


def _frame_counts(traj_paths, top, wd, env):
    """Frames in each trajectory file, so a global index can be split per file."""
    counts = []
    for t in traj_paths:
        out, _err, _code = safe_run(
            f"{amber_bin('cpptraj')} -p {top} -y {t} -tl", wd, env)
        n = None
        for line in out.splitlines():
            if "Frames:" in line:
                try:
                    n = int(line.split("Frames:")[1].split()[0])
                except (IndexError, ValueError):
                    n = None
        if n is None:
            return None
        counts.append(n)
    return counts


def _interface_mask(system):
    """cpptraj mask for interface heavy atoms, used as the clustering metric."""
    resids = sorted(ID.get_mask(system, dasa_cutoff=0.5, polar=False) or [])
    if not resids:
        return None
    return ":" + ",".join(str(r) for r in resids) + "&!@H="


def _parse_summary(path):
    """Largest cluster from a cpptraj cluster summary: (n_frames, centroid_frame).

    Columns: #Cluster Frames Frac AvgDist Stdev Centroid AvgCDist. The largest
    cluster is selected by frame count, independent of row order.
    """
    best = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 6:
            continue
        try:
            n_frames, centroid = int(f[1]), int(f[5])
        except ValueError:
            continue
        if best is None or n_frames > best[0]:
            best = (n_frames, centroid)
    return best


def _global_to_file(counts, frame):
    """Map a 1-based global frame index to (file position, 1-based local index)."""
    remaining = frame
    for i, n in enumerate(counts):
        if remaining <= n:
            return i, remaining
        remaining -= n
    return None, None


def process(system, rep, out_base, force=False):
    env, _ = amber_env()
    sysdir = REP_DIRS[rep] / system
    top = (sysdir / f"{system}_complex_wat.top").resolve()
    trajs = [(sysdir / f"{system}_complex_wat.{p}.mdcrd").resolve() for p in PRODS]
    if not top.exists() or not all(t.exists() for t in trajs):
        return system, rep, "missing top/traj", None

    out_dir = Path(out_base).resolve() / system / "ff14sb" / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_dst, wet_dst = out_dir / "dry_f1.pdb", out_dir / "wet_f1.pdb"
    info_f = out_dir / "cluster_info.json"
    if not force and dry_dst.exists() and wet_dst.exists() and info_f.exists():
        return system, rep, "cached", json.loads(info_f.read_text())

    mask = _interface_mask(system)
    if not mask:
        return system, rep, "no interface mask", None
    water_mask = ID.polar_mask_for_waters(system)
    if not water_mask:
        return system, rep, "no polar dASA mask", None

    work = out_dir / "_cluster"
    work.mkdir(exist_ok=True)

    counts = _frame_counts(trajs, top, work, env)
    if not counts:
        return system, rep, "frame count failed", None

    # Solvent is stripped before clustering; the RMSD uses interface heavy atoms only.
    trajin = "".join(f"trajin {t}\n" for t in trajs)
    _cpptraj(
        f"{trajin}"
        "autoimage\n"
        "strip :WAT,Cl-,Na+\n"
        f"rms first {mask}\n"
        f"cluster hieragglo clusters {N_CLUSTERS} rms {mask} "
        "summary summary.dat info info.dat\n"
        "run\nquit\n",
        top, work, env, "clust")

    summary = work / "summary.dat"
    if not summary.exists():
        return system, rep, "clustering failed", None
    best = _parse_summary(summary)
    if best is None:
        return system, rep, "no clusters parsed", None
    n_frames, centroid = best
    total = sum(counts)
    idx, local = _global_to_file(counts, centroid)
    if idx is None:
        return system, rep, f"centroid {centroid} out of range", None
    traj, fr = trajs[idx], local

    # Same dry/wet extraction commands as extract_snapshots.py.
    _cpptraj(
        f"trajin {traj} {fr} {fr}\n"
        "autoimage\n"
        "strip :WAT,Cl-,Na+\n"
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
        "system": system, "rep": rep,
        "prods": list(PRODS), "total_frames": total,
        "cluster_frames": n_frames,
        "cluster_population": round(n_frames / total, 4),
        "centroid_global_frame": centroid,
        "source_traj": traj.name, "source_local_frame": fr,
        "n_clusters": N_CLUSTERS, "rms_mask": mask,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, rep, "ok", info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--reps", nargs="*", default=sorted(REP_DIRS))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(s, r) for s in args.systems for r in args.reps]
    print(f"clustering {len(jobs)} (system, replica) pairs on {PRODS} "
          f"with {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, s, r, args.out_base, args.force): (s, r)
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


def rep_pad(r):
    return f"{r:6}"


if __name__ == "__main__":
    main()
