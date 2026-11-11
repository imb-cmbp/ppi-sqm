#!/usr/bin/env python3
"""Nwat-MMGBSA (igb=8, Nwat=30, polar closest) on cluster-centroid frames.

Scores one frame per basin listed in <snap-base>/<SYS>/ff14sb/<rep>/basins.json,
with the same settings as the trajectory-based Nwat-MMGBSA. Writes
detailed_centroids.csv and population-weighted / unweighted per-replica scores
(mmgbsa_weighted.csv, mmgbsa_unweighted.csv) to --out-base.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import AMBERHOME, WORK  # noqa: E402

REPO = WORK

from cluster_extract import traj_paths  # noqa: E402
from cluster_representatives import REP_DIRS, _cpptraj, _frame_counts, _global_to_file  # noqa: E402
import interface_dasa as ID  # noqa: E402
from amber_setup import amber_env  # noqa: E402
from importlib.util import spec_from_file_location, module_from_spec  # noqa: E402

_spec = spec_from_file_location(
    "nwat_mmgbsa", Path(__file__).resolve().parent / "nwat_mmgbsa.py")
_nwat = module_from_spec(_spec)
_spec.loader.exec_module(_nwat)
load_features = _nwat.load_features
parse_delta = _nwat.parse_delta
_water_resrange = _nwat._water_resrange

NWAT = 30
SNAP_BASE = REPO / "screening_ff_md_sqm_clust_aa_gt5"
OUT_BASE = REPO / "md" / "mmgbsa_clust_aa_gt5"


def _score_one(system, rep, snap, global_frame, out_base, nwat, force, sysdir=None):
    feat = load_features()[system]
    rec, lig, last = feat["rec"], feat["lig"], feat["last"]
    intmask = ID.polar_mask_for_waters(system)
    if not intmask:
        return system, rep, snap, None, "no polar mask"

    wd = Path(out_base) / system / rep / f"f{snap}"
    wd.mkdir(parents=True, exist_ok=True)
    fin = wd / f"FINAL_RESULTS_CLOSEST{nwat}"
    if not force and fin.exists():
        dg = parse_delta(fin.read_text())
        if dg is not None:
            return system, rep, snap, dg, "cached"

    env, _ = amber_env()
    _root, top, trajs = traj_paths(system, rep, sysdir)
    if not top.exists() or not all(t.exists() for t in trajs):
        return system, rep, snap, None, "missing top/traj"
    counts = _frame_counts(trajs, top, wd, env)
    if not counts:
        return system, rep, snap, None, "frame count failed"
    idx, local = _global_to_file(counts, global_frame)
    if idx is None:
        return system, rep, snap, None, f"frame {global_frame} oob"
    traj = trajs[idx]

    ctop = wd / f"nwat{nwat}.{system}.top"
    rtop = wd / f"nwat{nwat}.{system}.rec.top"
    ltop = wd / "ligand.top"
    mdc = wd / f"nwat{nwat}.{system}.mdcrd"
    for f in wd.glob("clo.*"):
        f.unlink(missing_ok=True)
    _cpptraj(
        f"trajin {traj} {local} {local}\n"
        "center @CA,C,N mass origin\n"
        "image origin center\n"
        "strip :Cl-,:Na+\n"
        f"closest {nwat} {intmask} noimage outprefix clo\n"
        f"trajout {mdc.name} nobox\nrun\nquit\n",
        top, wd, env, "cmplx")
    gen = list(wd.glob("clo.*"))
    if not gen or not mdc.exists():
        return system, rep, snap, None, "closest failed"
    _cpptraj(
        f"parmbox nobox\nparmwrite out {ctop.name} amber\n"
        f"parmstrip {lig}\nparmwrite out {rtop.name} amber\nrun\nquit\n",
        gen[0], wd, env, "topsplit")
    _cpptraj(
        f"parmbox nobox\nparmstrip {rec},:WAT\n"
        f"parmwrite out {ltop.name} amber\nrun\nquit\n",
        gen[0], wd, env, "lig")
    if not (ctop.exists() and rtop.exists() and ltop.exists()):
        return system, rep, snap, None, "top split failed"
    try:
        w, a = _water_resrange(ctop, nwat, last)
    except Exception as e:
        return system, rep, snap, None, f"water range: {e}"
    recmask = f"{rec}:{w}-{a}"
    (wd / f"mmpbsa_closest_nwat{nwat}.in").write_text(
        f"&general\nreceptor_mask={recmask}, ligand_mask={lig}, "
        f"startframe=1, endframe=1, interval=1, verbose=1,\n/\n"
        f"&gb\nigb=8, saltcon=0.15,\n/\n")
    import subprocess
    mmcmd = (
        f"source {AMBERHOME}/amber.sh && MMPBSA.py -O "
        f"-i mmpbsa_closest_nwat{nwat}.in -o FINAL_RESULTS_CLOSEST{nwat} "
        f"-cp {ctop.name} -rp {rtop.name} -lp {ltop.name} -y {mdc.name}")
    subprocess.run(["bash", "-c", mmcmd], cwd=wd, env=env,
                   capture_output=True, text=True)
    for junk in wd.glob("_MMPBSA_*"):
        try:
            junk.unlink()
        except OSError:
            pass
    if fin.exists():
        dg = parse_delta(fin.read_text())
        if dg is not None:
            return system, rep, snap, dg, "ok"
    return system, rep, snap, None, "mmpbsa failed"


def collect_jobs(snap_base):
    jobs = []
    for info_f in sorted(Path(snap_base).glob("*/ff14sb/*/basins.json")):
        info = json.loads(info_f.read_text())
        for b in info["basins"]:
            jobs.append((info["system"], info["rep"], b["snap"],
                         b.get("centroid_global_frame") or b.get("centroid")))
    return jobs


def write_outputs(rows, snap_base, out_base, nwat):
    """rows: list of (system, rep, snap, dg, status)."""
    by = {(s, r, n): dg for s, r, n, dg, _st in rows if dg is not None}
    # out_base is not created by any worker when there are no jobs.
    Path(out_base).mkdir(parents=True, exist_ok=True)
    detailed = Path(out_base) / "detailed_centroids.csv"
    with detailed.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["System", "rep", "snap", "frac", "dG", "status"])
        for info_f in sorted(Path(snap_base).glob("*/ff14sb/*/basins.json")):
            info = json.loads(info_f.read_text())
            for b in info["basins"]:
                key = (info["system"], info["rep"], b["snap"])
                dg = by.get(key)
                st = next((st for s, r, n, _d, st in rows
                           if (s, r, n) == key), "")
                w.writerow([info["system"], info["rep"], b["snap"],
                            b["frac"], "" if dg is None else dg, st])

    # per-replica weighted / unweighted, same shape as mmgbsa_results_*.csv
    for kind, path in (("weighted", Path(out_base) / "mmgbsa_weighted.csv"),
                       ("unweighted", Path(out_base) / "mmgbsa_unweighted.csv")):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["System", "rep", "Nwat", "dG", "n_basins", "covered"])
            for info_f in sorted(Path(snap_base).glob("*/ff14sb/*/basins.json")):
                info = json.loads(info_f.read_text())
                num = den = 0.0
                vals = []
                for b in info["basins"]:
                    dg = by.get((info["system"], info["rep"], b["snap"]))
                    if dg is None:
                        continue
                    vals.append(dg)
                    num += float(b["frac"]) * dg
                    den += float(b["frac"])
                if not vals:
                    continue
                score = (num / den) if kind == "weighted" else (sum(vals) / len(vals))
                w.writerow([info["system"], info["rep"], nwat, f"{score:.4f}",
                            len(vals), f"{den:.4f}"])
    return detailed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snap-base", default=str(SNAP_BASE))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--nwat", type=int, default=NWAT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--reps", nargs="*")
    ap.add_argument("--sysdir", default=None,
                    help="MD directory containing <SYS>/<SYS>_complex_wat.prod*.mdcrd")
    args = ap.parse_args()

    jobs = collect_jobs(args.snap_base)
    if args.systems:
        jobs = [j for j in jobs if j[0] in args.systems]
    if args.reps:
        jobs = [j for j in jobs if j[1] in args.reps]
    print(f"MM-GBSA Nwat={args.nwat} on {len(jobs)} centroids, "
          f"workers={args.workers}", flush=True)
    rows = []
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_score_one, s, r, n, fr, args.out_base,
                            args.nwat, args.force, args.sysdir): (s, r, n)
                for s, r, n, fr in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            system, rep, snap, dg, status = fut.result()
            rows.append((system, rep, snap, dg, status))
            if dg is not None:
                n_ok += 1
            print(f"  [{k}/{len(jobs)}] {system} {rep} f{snap}: "
                  f"{status} dG={dg}", flush=True)
    write_outputs(rows, args.snap_base, args.out_base, args.nwat)
    print(f"\n{n_ok}/{len(jobs)} scored -> {args.out_base}", flush=True)


if __name__ == "__main__":
    main()
