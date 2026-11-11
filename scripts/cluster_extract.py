"""Shared frame extraction for cluster centroids (dry + closest-N wet)."""
from __future__ import annotations

from pathlib import Path

from cluster_representatives import REP_DIRS, _cpptraj, _frame_counts, _global_to_file
import interface_dasa as ID
from common import sanitize_wet_pdb_to_hoh
from amber_setup import amber_env

PRODS = ("prod1", "prod2", "prod3", "prod4")
MAX_WATERS = 30
STRIP_SOLVENT = ":WAT,HOH,T4D,Na+,Cl-"


def parse_all_clusters(path):
    """Return list of dicts: cluster, n_frames, frac, centroid (1-based global)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 6:
            continue
        try:
            rows.append({
                "cluster": int(f[0]),
                "n_frames": int(f[1]),
                "frac": float(f[2]),
                "centroid": int(f[5]),
            })
        except ValueError:
            continue
    rows.sort(key=lambda r: -r["n_frames"])
    return rows


def traj_paths(system, rep, sysdir=None):
    root = Path(sysdir) / system if sysdir else REP_DIRS[rep] / system
    top = (root / f"{system}_complex_wat.top").resolve()
    trajs = [(root / f"{system}_complex_wat.{p}.mdcrd").resolve() for p in PRODS]
    return root, top, trajs


def extract_frame(system, rep, global_frame, dry_dst, wet_dst, sysdir=None):
    """Write dry/wet PDBs for a 1-based global production-frame index."""
    env, _ = amber_env()
    _root, top, trajs = traj_paths(system, rep, sysdir)
    if not top.exists() or not all(t.exists() for t in trajs):
        return False, "missing top/traj"
    water_mask = ID.polar_mask_for_waters(system)
    if not water_mask:
        return False, "no polar dASA mask"
    work = Path(dry_dst).resolve().parent / "_extract"
    work.mkdir(parents=True, exist_ok=True)
    counts = _frame_counts(trajs, top, work, env)
    if not counts:
        return False, "frame count failed"
    idx, local = _global_to_file(counts, global_frame)
    if idx is None:
        return False, f"frame {global_frame} out of range"
    traj, fr = trajs[idx], local
    dry_dst, wet_dst = Path(dry_dst), Path(wet_dst)

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
    if dry_dst.exists() and wet_dst.exists():
        return True, {"source_traj": traj.name, "source_local_frame": fr,
                      "total_frames": sum(counts)}
    return False, "extraction failed"
