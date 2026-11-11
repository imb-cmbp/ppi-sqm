#!/usr/bin/env python3
"""Select scoring frames from an Amber topology and its production trajectories.

Two schemes match the paper:

  even-10   ten evenly spaced frames of the final production nanosecond
            (prod4, frames 1, 101, ..., 901 of a 1000-frame file)

  basin     average-linkage hierarchical clustering of all production frames
            into ten clusters; the centroid of every basin that holds more
            than --min-frac (default 0.05) of the frames is kept. The distance
            is RMSD over all-atom, backbone or interface side-chain atoms, or
            the Jaccard distance of interchain residue contacts.

Each selected frame is written as dry_fN.pdb (no solvent) and wet_fN.pdb (the
--max-waters waters closest to the interface water mask). A basins.json file
records the basin populations for population-weighted scoring.

Examples
--------
python scripts/select_frames.py --scheme even10 --sysdir $WORK/md/systems_propka_r1 \\
    --rep r1 --out-base $WORK/even10_r1 --systems 1ACB
python scripts/select_frames.py --scheme basin --metric aa --sysdir ... --rep r1 \\
    --out-base $WORK/basins_aa_r1
python scripts/select_frames.py --scheme basin --metric iface_sc --sysdir ... \\
    --dasa-resids data/interface_masks/dasa0.5/1ACB.json --systems 1ACB
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amber_setup import amber_bin, amber_env  # noqa: E402
from common import PDB_IDS, sanitize_wet_pdb_to_hoh, safe_run  # noqa: E402
from cluster_extract import parse_all_clusters  # noqa: E402
from cluster_representatives import _cpptraj, _frame_counts, _global_to_file  # noqa: E402
import interface_dasa as ID  # noqa: E402
import truncate_interface as TI  # noqa: E402

PRODS = ("prod1", "prod2", "prod3", "prod4")
STRIP_SOLVENT = ":WAT,HOH,T4D,Na+,Cl-"
RMS_MASKS = {
    "aa": "!(:WAT,HOH,T4D,Na+,Cl-)&!@H=",
    "bb": "!(:WAT,HOH,T4D,Na+,Cl-)&@CA,C,N",
}
CONTACT_CUT = 4.5


def _traj_paths(sysdir, system):
    root = Path(sysdir) / system
    top = (root / f"{system}_complex_wat.top").resolve()
    trajs = [(root / f"{system}_complex_wat.{p}.mdcrd").resolve() for p in PRODS]
    return root, top, trajs


def _parse_dasa_arg(value):
    """cpptraj mask, JSON file with 'resids', or None."""
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return sorted(json.loads(path.read_text())["resids"])
    resids = set()
    for part in value.lstrip(":").split(","):
        lo, _, hi = part.strip().partition("-")
        if lo:
            resids.update(range(int(lo), int(hi or lo) + 1))
    return sorted(resids)


def _water_mask(system, dasa_resids, polar_waters, water_mask):
    """cpptraj selection string used by ``closest`` for the wet PDB."""
    if water_mask:
        return water_mask if water_mask.startswith(":") else f":{water_mask.lstrip(':')}"
    if dasa_resids is not None:
        resids = dasa_resids
        if polar_waters:
            # Keep only polar residue numbers when the user gave a full mask;
            # without residue names we cannot filter, so use the shipped polar cache.
            polar = ID.get_mask(system, dasa_cutoff=0.5, polar=True) or set()
            if polar:
                resids = [r for r in resids if r in polar] or resids
        return ":" + ",".join(str(r) for r in resids)
    if polar_waters:
        mask = ID.polar_mask_for_waters(system)
        if mask:
            return mask
    resids = sorted(ID.get_mask(system, dasa_cutoff=0.5, polar=polar_waters) or [])
    if not resids:
        raise SystemExit(
            f"{system}: no interface water mask. Pass --water-mask, --dasa-resids, "
            "or put a JSON mask under data/interface_masks/")
    return ":" + ",".join(str(r) for r in resids)


def _iface_rms_mask(system, dasa_resids, mode="iface_sc"):
    resids = dasa_resids or sorted(ID.get_mask(system, dasa_cutoff=0.5, polar=False) or [])
    if not resids:
        return None
    sel = ":" + ",".join(str(r) for r in resids)
    if mode == "iface_sc":
        return f"{sel}&!@H=&!@C,CA,N,O"
    if mode == "iface_heavy":
        return f"{sel}&!@H="
    if mode == "iface_polar":
        return f"{sel}&!@H=&!@CA,C"
    raise ValueError(mode)


def extract_frame(top, trajs, counts, global_frame, dry_dst, wet_dst, water_mask,
                  max_waters, env, work):
    """Write dry and wet PDBs for a 1-based global frame index over ``trajs``."""
    idx, local = _global_to_file(counts, global_frame)
    if idx is None:
        return False, f"frame {global_frame} out of range"
    traj, fr = trajs[idx], local
    dry_dst, wet_dst = Path(dry_dst), Path(wet_dst)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

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
        f"closest {max_waters} {water_mask} noimage outprefix clo\n"
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


def even10(system, rep, sysdir, out_base, ff_folder, n_frames, stride, max_waters,
           water_mask, force):
    env, _ = amber_env()
    _root, top, trajs = _traj_paths(sysdir, system)
    traj = trajs[-1]  # prod4, the final nanosecond
    if not top.exists() or not traj.exists():
        return system, "missing top/traj", None
    out_dir = Path(out_base) / system / ff_folder / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    info_f = out_dir / "basins.json"
    if not force and info_f.exists() and all(
            (out_dir / f"wet_f{i}.pdb").exists() for i in range(1, n_frames + 1)):
        return system, "cached", json.loads(info_f.read_text())

    work = out_dir / "_extract"
    work.mkdir(exist_ok=True)
    counts = _frame_counts([traj], top, work, env)
    if not counts:
        return system, "frame count failed", None
    # Global indices over prod4 alone.
    basins = []
    for i in range(1, n_frames + 1):
        fr = 1 + (i - 1) * stride
        dry = out_dir / f"dry_f{i}.pdb"
        wet = out_dir / f"wet_f{i}.pdb"
        ok, extra = extract_frame(top, [traj], counts, fr, dry, wet, water_mask,
                                  max_waters, env, work / f"f{i}")
        if not ok:
            return system, f"extract f{i} failed: {extra}", None
        basins.append({"snap": i, "frac": 1.0 / n_frames,
                       "centroid_global_frame": fr, **extra})
    info = {
        "system": system, "rep": rep, "ff": ff_folder, "scheme": "even10",
        "n_frames": n_frames, "stride": stride, "max_waters": max_waters,
        "water_mask": water_mask, "n_basins": len(basins),
        "covered_frac": 1.0, "basins": basins,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, "ok", info


def cluster_rmsd(system, rep, sysdir, out_base, ff_folder, metric, n_clusters,
                 min_frac, max_waters, water_mask, dasa_resids, force):
    env, _ = amber_env()
    _root, top, trajs = _traj_paths(sysdir, system)
    if not top.exists() or not all(t.exists() for t in trajs):
        return system, "missing top/traj", None
    if metric in RMS_MASKS:
        rms_mask = RMS_MASKS[metric]
    else:
        rms_mask = _iface_rms_mask(system, dasa_resids, metric)
        if not rms_mask:
            return system, "no interface mask for RMSD", None

    out_dir = Path(out_base) / system / ff_folder / rep
    work = out_dir / "_cluster"
    work.mkdir(parents=True, exist_ok=True)
    summary = work / "summary.dat"
    info_f = out_dir / "basins.json"
    if not force and summary.exists() and info_f.exists():
        cached = json.loads(info_f.read_text())
        if cached.get("basins") and all(
                (out_dir / f"wet_f{b['snap']}.pdb").exists() for b in cached["basins"]):
            return system, "cached", cached

    counts = _frame_counts(trajs, top, work, env)
    if not counts:
        return system, "frame count failed", None
    trajin = "".join(f"trajin {t}\n" for t in trajs)
    out, err, _code = _cpptraj(
        f"{trajin}"
        "autoimage\n"
        f"strip {STRIP_SOLVENT}\n"
        f"rms first {rms_mask}\n"
        f"cluster hieragglo clusters {n_clusters} rms {rms_mask} "
        "summary summary.dat info info.dat\n"
        "run\nquit\n",
        top, work, env, "clust")
    if not summary.exists():
        return system, f"clustering failed {(err or out or '')[-400:]}", None

    kept = [c for c in parse_all_clusters(summary) if c["frac"] >= min_frac]
    if not kept:
        return system, "no basin above cutoff", None
    basins = []
    for i, c in enumerate(kept, 1):
        dry = out_dir / f"dry_f{i}.pdb"
        wet = out_dir / f"wet_f{i}.pdb"
        ok, extra = extract_frame(top, trajs, counts, c["centroid"], dry, wet,
                                  water_mask, max_waters, env, work / f"f{i}")
        if not ok:
            return system, f"extract f{i} failed: {extra}", None
        basins.append({
            "snap": i, "cluster": c["cluster"], "n_frames": c["n_frames"],
            "frac": c["frac"], "centroid_global_frame": c["centroid"],
            **(extra if isinstance(extra, dict) else {}),
        })
    info = {
        "system": system, "rep": rep, "ff": ff_folder, "scheme": "basin",
        "metric": metric, "rms_mask": rms_mask, "n_clusters": n_clusters,
        "min_frac": min_frac, "max_waters": max_waters, "water_mask": water_mask,
        "total_frames": sum(counts), "n_basins": len(basins),
        "covered_frac": round(sum(b["frac"] for b in basins), 4),
        "basins": basins,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, "ok", info


def _load_contact_series(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(x) for x in parts[1:]])
    if not rows:
        return None
    return (np.asarray(rows) > 0.5).astype(np.uint8)


def _medoid(X, members):
    sub = X[members]
    if len(members) == 1:
        return int(members[0])
    d = squareform(pdist(sub, metric="jaccard"))
    return int(members[int(np.argmin(d.mean(axis=1)))])


def cluster_contacts(system, rep, sysdir, out_base, ff_folder, n_clusters,
                     min_frac, max_waters, water_mask, n_chain_a, force):
    env, _ = amber_env()
    _root, top, trajs = _traj_paths(sysdir, system)
    if not top.exists() or not all(t.exists() for t in trajs):
        return system, "missing top/traj", None
    if n_chain_a is None or n_chain_a < 2:
        return system, "contacts metric needs --n-chain-a or --input-pdb", None

    out_dir = Path(out_base) / system / ff_folder / rep
    work = out_dir / "_cluster"
    work.mkdir(parents=True, exist_ok=True)
    info_f = out_dir / "basins.json"
    if not force and info_f.exists() and all(
            (out_dir / f"wet_f{b['snap']}.pdb").exists()
            for b in json.loads(info_f.read_text()).get("basins", [])):
        return system, "cached", json.loads(info_f.read_text())

    series = work / "ic_series.dat"
    trajin = "".join(f"trajin {t}\n" for t in trajs)
    _cpptraj(
        f"{trajin}"
        "autoimage\n"
        f"strip {STRIP_SOLVENT}\n"
        f"nativecontacts name IC :1-{n_chain_a}&!@H= :{n_chain_a + 1}-9999&!@H= "
        f"distance {CONTACT_CUT} byresidue skipnative savenonnative series "
        f"seriesnnout {series.name}\n"
        "run\nquit\n",
        top, work, env, "contacts")
    X = _load_contact_series(series)
    if X is None or X.shape[0] < 10:
        return system, f"empty contact series n_chain_a={n_chain_a}", None
    vary = X.std(axis=0) > 0
    Xv = X[:, vary] if vary.any() else X
    if Xv.shape[1] < 2:
        return system, "fewer than 2 unique contacts", None
    labels = fcluster(linkage(pdist(Xv, metric="jaccard"), method="average"),
                      t=n_clusters, criterion="maxclust")
    raw = []
    n = len(labels)
    for cid in sorted(set(labels), key=lambda c: -np.sum(labels == c)):
        members = np.where(labels == cid)[0]
        raw.append({
            "cluster": int(cid), "n_frames": int(len(members)),
            "frac": round(float(len(members) / n), 4),
            "centroid": int(_medoid(Xv, members) + 1),
        })
    kept = [b for b in raw if b["frac"] >= min_frac]
    if not kept:
        return system, "no basin above cutoff", None

    counts = _frame_counts(trajs, top, work, env)
    basins = []
    for i, c in enumerate(kept, 1):
        dry = out_dir / f"dry_f{i}.pdb"
        wet = out_dir / f"wet_f{i}.pdb"
        ok, extra = extract_frame(top, trajs, counts, c["centroid"], dry, wet,
                                  water_mask, max_waters, env, work / f"f{i}")
        if not ok:
            return system, f"extract f{i} failed: {extra}", None
        basins.append({
            "snap": i, **c, "centroid_global_frame": c["centroid"],
            **(extra if isinstance(extra, dict) else {}),
        })
    info = {
        "system": system, "rep": rep, "ff": ff_folder, "scheme": "basin",
        "metric": f"interchain_residue_contacts_{CONTACT_CUT}A_jaccard",
        "n_chain_a": n_chain_a, "n_clusters": n_clusters, "min_frac": min_frac,
        "max_waters": max_waters, "water_mask": water_mask,
        "n_contacts": int(Xv.shape[1]), "total_frames": n,
        "n_basins": len(basins),
        "covered_frac": round(sum(b["frac"] for b in basins), 4),
        "basins": basins,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, "ok", info


def _n_chain_a_for(system, sysdir, n_chain_a, input_pdb):
    if n_chain_a is not None:
        return n_chain_a
    if input_pdb is not None:
        dry = Path(sysdir) / system / f"{system}_complex_wat.top"
        # Prefer a dry snapshot if one already exists next to an older run;
        # otherwise scale from the input PDB using residue count of the topology
        # is not available as a PDB. Fall back to the input-PDB chain length.
        from common import get_first_chain_nres
        return get_first_chain_nres(input_pdb)
    # Optional: tleap capfix next to the topology (paper MD layout).
    from extract_bridging_waters import n_chain_a_tleap
    return n_chain_a_tleap(system, sysdir)


def process(system, args, water_mask, dasa_resids):
    nA = _n_chain_a_for(system, args.sysdir, args.n_chain_a, args.input_pdb)
    if args.scheme == "even10":
        return even10(system, args.rep, args.sysdir, args.out_base, args.ff_folder,
                      args.n_frames, args.stride, args.max_waters, water_mask,
                      args.force)
    if args.metric == "contacts":
        return cluster_contacts(
            system, args.rep, args.sysdir, args.out_base, args.ff_folder,
            args.n_clusters, args.min_frac, args.max_waters, water_mask, nA,
            args.force)
    return cluster_rmsd(
        system, args.rep, args.sysdir, args.out_base, args.ff_folder,
        args.metric, args.n_clusters, args.min_frac, args.max_waters,
        water_mask, dasa_resids, args.force)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=["even10", "basin"], required=True)
    ap.add_argument("--metric",
                    choices=["aa", "bb", "iface_sc", "iface_heavy", "iface_polar",
                             "contacts"],
                    default="aa",
                    help="basin distance (ignored for even10); default aa")
    ap.add_argument("--sysdir", required=True,
                    help="directory of <ID>/<ID>_complex_wat.top and .prod1-4.mdcrd")
    ap.add_argument("--rep", default="r1", help="replica label written into the output tree")
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--ff-folder", default="ff14sb")
    ap.add_argument("--n-frames", type=int, default=10, help="even10: number of frames")
    ap.add_argument("--stride", type=int, default=100, help="even10: frame stride")
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="keep basins with at least this fraction of frames")
    ap.add_argument("--max-waters", type=int, default=30,
                    help="waters kept in each wet PDB (cpptraj closest)")
    ap.add_argument("--polar-waters", action="store_true", default=True,
                    help="anchor waters on polar interface residues (default)")
    ap.add_argument("--any-waters", action="store_true",
                    help="anchor waters on every interface residue")
    ap.add_argument("--water-mask", default=None,
                    help="cpptraj mask for closest, e.g. :37,38,40; overrides dASA")
    ap.add_argument("--dasa-resids", default=None,
                    help="interface residues for iface_* metrics and, if no "
                         "--water-mask, for water selection: cpptraj mask or JSON")
    ap.add_argument("--n-chain-a", type=int, default=None,
                    help="residues in partner A (required for contacts unless "
                         "--input-pdb or a capfix.pdb is available)")
    ap.add_argument("--input-pdb", default=None,
                    help="complex PDB whose first chain defines partner A")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.any_waters:
        args.polar_waters = False
    if args.scheme == "basin" and args.metric is None:
        ap.error("--metric is required for --scheme basin")

    dasa_resids = _parse_dasa_arg(args.dasa_resids)
    input_pdb = Path(args.input_pdb) if args.input_pdb else None
    args.input_pdb = input_pdb

    jobs = []
    for system in args.systems:
        try:
            water_mask = _water_mask(system, dasa_resids, args.polar_waters,
                                     args.water_mask)
        except SystemExit as exc:
            print(f"  {system}: {exc}", flush=True)
            continue
        jobs.append((system, water_mask))

    print(f"{args.scheme}"
          + (f"/{args.metric}" if args.scheme == "basin" else "")
          + f" on {len(jobs)} systems, max_waters={args.max_waters}, "
            f"polar_waters={args.polar_waters}", flush=True)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process, system, args, water_mask, dasa_resids): system
            for system, water_mask in jobs
        }
        for k, fut in enumerate(as_completed(futs), 1):
            system, status, info = fut.result()
            nb = info["n_basins"] if info else 0
            print(f"  [{k}/{len(jobs)}] {system}: {status} n={nb}", flush=True)
            if status in ("ok", "cached"):
                n_ok += 1
    print(f"{n_ok}/{len(jobs)} done -> {args.out_base}")


if __name__ == "__main__":
    main()
