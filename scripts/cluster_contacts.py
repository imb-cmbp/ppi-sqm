#!/usr/bin/env python3
"""Cluster production frames by interchain residue-contact pattern.

Metric: Jaccard distance of the by-residue contact bit-vector between the two
partners (any heavy-atom pair within 4.5 A). Native-contact bias is turned off
(skipnative) so contacts that form later in the trajectory still count.

Keeps every basin with population >= --min-frac, writes the medoid of each as
wet/dry_f1..fK plus basins.json (same layout as extract_cluster_basins.py).
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
from paths import CHAIN_REF, WORK  # noqa: E402

REPO = WORK

from cluster_extract import extract_frame, traj_paths, PRODS  # noqa: E402
from cluster_representatives import REP_DIRS, _cpptraj  # noqa: E402
from extract_bridging_waters import n_chain_a_tleap  # noqa: E402
from common import get_first_chain_nres  # noqa: E402
from common import PDB_IDS  # noqa: E402
from amber_setup import amber_env  # noqa: E402

OUT_BASE = REPO / "screening_ff_md_sqm_clust_contacts"
PDB_INPUT = CHAIN_REF
MIN_FRAC = 0.05
N_CLUSTERS = 10
CUT = 4.5


def _load_series(path):
    """cpptraj seriesnnout: '#Frame c1 c2 ...' then one row per frame."""
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            rows.append([float(x) for x in parts[1:]])
    if not rows:
        return None
    return (np.asarray(rows) > 0.5).astype(np.uint8)


def _medoid(X, members):
    """Index (into the full matrix) of the member closest to the others."""
    sub = X[members]
    if len(members) == 1:
        return int(members[0])
    d = squareform(pdist(sub, metric="jaccard"))
    return int(members[int(np.argmin(d.mean(axis=1)))])


def cluster_contacts(system, rep, sysdir, work, n_clusters, cut):
    env, _ = amber_env()
    _root, top, trajs = traj_paths(system, rep, sysdir)
    if not top.exists() or not all(t.exists() for t in trajs):
        return None, "missing top/traj"
    # Chain-A boundary in tleap numbering (from capfix.pdb), which matches the MD
    # topology. The protonated input PDB keeps crystal residue gaps (e.g. 1ZHI,
    # 3BZD) and would assign all residues to partner A.
    nA = (n_chain_a_tleap(system, REPO / "md" / "systems")
          or get_first_chain_nres(PDB_INPUT / f"{system}_input.pdb"))
    if nA < 2:
        return None, "bad chain split"
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    series = work / "ic_series.dat"
    trajin = "".join(f"trajin {t}\n" for t in trajs)
    _cpptraj(
        f"{trajin}"
        "autoimage\n"
        "strip :WAT,HOH,T4D,Na+,Cl-\n"
        f"nativecontacts name IC :1-{nA}&!@H= :{nA + 1}-9999&!@H= "
        f"distance {cut} byresidue skipnative savenonnative series "
        f"seriesnnout {series.name}\n"
        "run\nquit\n",
        top, work, env, "contacts")
    X = _load_series(series)
    if X is None or X.shape[0] < 10:
        return None, f"empty contact series nA={nA} ({series})"
    if X.shape[1] < 2:
        return None, "fewer than 2 unique contacts"
    # Constant columns do not affect Jaccard; drop them to save time.
    vary = X.std(axis=0) > 0
    Xv = X[:, vary] if vary.any() else X
    dist = pdist(Xv, metric="jaccard")
    Z = linkage(dist, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    basins = []
    n = len(labels)
    for cid in sorted(set(labels), key=lambda c: -np.sum(labels == c)):
        members = np.where(labels == cid)[0]
        frac = len(members) / n
        med = _medoid(Xv, members)
        basins.append({
            "cluster": int(cid),
            "n_frames": int(len(members)),
            "frac": round(float(frac), 4),
            "centroid": int(med + 1),  # 1-based global frame
        })
    return {"n_frames": n, "n_contacts": int(Xv.shape[1]), "basins": basins}, None


def process(system, rep, out_base, ff_folder, min_frac, n_clusters, sysdir, force):
    out_dir = Path(out_base) / system / ff_folder / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    info_f = out_dir / "basins.json"
    if not force and info_f.exists():
        info = json.loads(info_f.read_text())
        if all((out_dir / f"wet_f{b['snap']}.pdb").exists() for b in info["basins"]):
            return system, rep, "cached", info

    raw, err = cluster_contacts(system, rep, sysdir, out_dir / "_cluster",
                                n_clusters, CUT)
    if raw is None:
        return system, rep, err, None
    kept = [b for b in raw["basins"] if b["frac"] >= min_frac]
    if not kept:
        return system, rep, "no basin above cutoff", raw

    basins = []
    for i, c in enumerate(kept, 1):
        dry = out_dir / f"dry_f{i}.pdb"
        wet = out_dir / f"wet_f{i}.pdb"
        ok, extra = extract_frame(system, rep, c["centroid"], dry, wet, sysdir)
        if not ok:
            return system, rep, f"extract f{i} failed: {extra}", None
        basins.append({
            "snap": i, **c,
            **({} if not isinstance(extra, dict) else extra),
        })
    info = {
        "system": system, "rep": rep, "ff": ff_folder,
        "metric": f"interchain_residue_contacts_{CUT}A_jaccard",
        "min_frac": min_frac, "n_clusters": n_clusters,
        "n_contacts": raw["n_contacts"],
        "n_chain_a": n_chain_a_tleap(system, REPO / "md" / "systems"),
        "total_frames": raw["n_frames"],
        "n_basins": len(basins),
        "covered_frac": round(sum(b["frac"] for b in basins), 4),
        "basins": basins,
    }
    info_f.write_text(json.dumps(info, indent=2))
    return system, rep, "ok", info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--reps", nargs="*", default=sorted(REP_DIRS))
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--ff-folder", default="ff14sb")
    ap.add_argument("--sysdir", default=None)
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC)
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = [(s, r) for s in args.systems for r in args.reps]
    print(f"interchain-contact clustering {len(jobs)} pairs, "
          f"keep frac>={args.min_frac}, k={args.n_clusters}", flush=True)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, s, r, args.out_base, args.ff_folder,
                            args.min_frac, args.n_clusters, args.sysdir,
                            args.force): (s, r)
                for s, r in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            system, rep, status, info = fut.result()
            nb = info["n_basins"] if info else 0
            cov = info.get("covered_frac", 0) if info else 0
            nc = info.get("n_contacts", 0) if info else 0
            print(f"  [{k}/{len(jobs)}] {system} {rep}: {status} "
                  f"n={nb} covered={cov:.2f} contacts={nc}", flush=True)
            if status in ("ok", "cached"):
                n_ok += 1
    print(f"\n{n_ok}/{len(jobs)} ok")


if __name__ == "__main__":
    main()
