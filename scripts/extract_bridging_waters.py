#!/usr/bin/env python3
"""Select interfacial waters that bridge both chains of a complex.

Each water is scored by max(distance to nearest chain-A heavy atom, distance
to nearest chain-B heavy atom), so only waters close to both partners rank
high. A pool of --pool-size waters closest to the polar dASA interface mask is
extracted with cpptraj (same centering, imaging and mask as extract_snapshots.py),
then re-ranked by this score and trimmed to --max-waters. Ions are excluded from
the chain atom pools. Writes <out-base>/<SYS>/ff14sb/<rep>/wet_f<i>.pdb.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHAIN_REF, WORK  # noqa: E402

REPO = WORK

from amber_setup import amber_bin, amber_env  # noqa: E402
from common import safe_run  # noqa: E402
from common import sanitize_wet_pdb_to_hoh  # noqa: E402
import interface_dasa as ID  # noqa: E402


def parse_atoms(pdb_path):
    """List of (chain, resseq, resname, name, x, y, z) for ATOM/HETATM lines,
    read from the fixed PDB columns."""
    out = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        resname = line[17:20].strip()
        name = line[12:16].strip()
        out.append((chain, resseq, resname, name, x, y, z))
    return out


def n_chain_a_tleap(system, sysdir):
    """Number of chain-A residues in tleap numbering (sequential from 1, caps
    included), read from <sys>_capfix.pdb, the tleap input of the MD topology.

    get_first_chain_nres() on the protonated input PDB gives a wrong chain
    boundary when a chain has an internal ACE/NME-capped gap (e.g. 1ZHI).
    Unlike the cpptraj-written wet PDB, capfix.pdb keeps the A/B chain IDs.
    """
    if sysdir is None:
        capfix = REPO / "md" / "systems" / system / f"{system}_capfix.pdb"
    else:
        capfix = Path(sysdir) / system / f"{system}_capfix.pdb"
    if not capfix.exists():
        # capfix.pdb is stored only in the base systems/ directory, not in
        # the per-replica directories.
        capfix = REPO / "md" / "systems" / system / f"{system}_capfix.pdb"
    if not capfix.exists():
        return None
    seen = set()
    for line in capfix.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != "A":
            continue
        seen.add(line[22:26])
    return len(seen) if seen else None

SYSDIR = REPO / "md" / "systems_propka_r1"
PDB_INPUT = CHAIN_REF
ION_RESNAMES = {"CL-", "NA+", "K+", "BR-", "I-", "CL", "NA", "MG", "ZN"}
N_FRAMES = 10
STRIDE = 100
POOL_SIZE = 120
MAX_WATERS = 30


def _cpptraj(inp, top, wd, env, tag):
    (wd / f"{tag}.cpptraj").write_text(inp)
    safe_run(f"{amber_bin('cpptraj')} -i {tag}.cpptraj -p {top}", wd, env)


def bridging_rerank(pdb_path, n_chain_a, max_waters, iface_resids=None):
    """Rewrite pdb_path in place, keeping the max_waters waters with the
    lowest bridging score. Returns (ok, n_waters_in_pool, n_kept).

    ``iface_resids``, if given, restricts the chain-A/B atom pools to residues
    of the dASA interface mask (same tleap residue numbering as the wet PDB).
    Without it, waters equidistant from non-interface atoms of both partners
    can also score well.
    """
    atoms = parse_atoms(pdb_path)
    prot_a, prot_b = [], []
    waters = {}  # resseq -> list of (name, x, y, z)
    for chain, resseq, resname, name, x, y, z in atoms:
        if resname in ("HOH", "WAT"):
            waters.setdefault(resseq, []).append((name, x, y, z))
            continue
        if resname.upper() in ION_RESNAMES or name.upper() in ("CL", "NA"):
            continue
        if name.startswith("H"):
            continue
        try:
            ri = int(resseq)
        except ValueError:
            continue
        if iface_resids is not None and ri not in iface_resids:
            continue
        (prot_a if ri <= n_chain_a else prot_b).append((x, y, z))

    if not waters or not prot_a or not prot_b:
        return False, 0, 0

    pa = np.array(prot_a)
    pb = np.array(prot_b)
    scored = []
    for resseq, wat_atoms in waters.items():
        ox = next(((x, y, z) for name, x, y, z in wat_atoms
                  if name.startswith("O")), None)
        if ox is None:
            continue
        p = np.array(ox)
        d_a = np.linalg.norm(pa - p, axis=1).min()
        d_b = np.linalg.norm(pb - p, axis=1).min()
        scored.append((max(d_a, d_b), resseq))
    scored.sort()
    keep = {r for _, r in scored[:max_waters]}

    lines = pdb_path.read_text(errors="ignore").splitlines()
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            if resname in ("HOH", "WAT") and resseq not in keep:
                continue
        out.append(line)
    pdb_path.write_text("\n".join(out) + "\n")
    return True, len(waters), len(keep)


def extract_one(system, out_base, n_frames, stride, pool_size, max_waters,
                sysdir_root, rep):
    env, _ = amber_env()
    sysdir = Path(sysdir_root).resolve() / system
    top = (sysdir / f"{system}_complex_wat.top").resolve()
    traj = (sysdir / f"{system}_complex_wat.prod4.mdcrd").resolve()
    if not top.exists() or not traj.exists():
        return system, 0, "missing top/traj"

    out_dir = Path(out_base).resolve() / system / "ff14sb" / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chain_a = n_chain_a_tleap(system, sysdir_root)
    if n_chain_a is None:
        return system, 0, "no capfix.pdb / chain A boundary"
    mask = ID.polar_mask_for_waters(system)
    if not mask:
        return system, 0, "no polar dASA mask"
    iface_resids = ID.get_mask(system, dasa_cutoff=0.5, polar=True)
    if not iface_resids:
        return system, 0, "empty polar dASA interface set"

    done = 0
    for i in range(1, n_frames + 1):
        fr = 1 + (i - 1) * stride
        wet_dst = out_dir / f"wet_f{i}.pdb"
        if wet_dst.exists():
            done += 1
            continue

        work = out_dir / f"_extract_f{i}"
        work.mkdir(exist_ok=True)
        for f in work.glob("clo.*"):
            f.unlink(missing_ok=True)
        _cpptraj(
            f"trajin {traj} {fr} {fr}\n"
            "center @CA,C,N mass origin\n"
            "image origin center\n"
            "strip :Na+,Cl-\n"
            f"closest {pool_size} {mask} noimage outprefix clo\n"
            f"trajout {wet_dst.name} nobox\nrun\nquit\n",
            top, work, env, "wet")
        # "outprefix clo" can also write a topology (clo.*.top); skip it,
        # since cpptraj reads zero frames from a .top file given as trajin.
        gen = [f for f in work.glob("clo.*") if not f.name.endswith(".top")]
        if gen:
            _cpptraj(f"trajin {gen[0].name} 1 1\n"
                     f"trajout {wet_dst.name}\nrun\nquit\n",
                     top, work, env, "wet2")

        w = work / wet_dst.name
        if w.exists():
            w.replace(wet_dst)
        for junk in work.glob("*"):
            junk.unlink(missing_ok=True)
        work.rmdir()

        if not wet_dst.exists():
            continue
        sanitize_wet_pdb_to_hoh(wet_dst)
        ok, n_pool, n_kept = bridging_rerank(wet_dst, n_chain_a, max_waters,
                                              iface_resids=iface_resids)
        if ok:
            done += 1

    return system, done, None if done == n_frames else f"only {done}/{n_frames}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--max-waters", type=int, default=MAX_WATERS)
    ap.add_argument("--sysdir", default=str(SYSDIR))
    ap.add_argument("--rep", default="r1")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    print(f"Bridging-aware extraction: pool={args.pool_size} -> "
          f"keep={args.max_waters}, {len(args.systems)} systems, rep={args.rep}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, s, args.out_base, args.n_frames,
                          args.stride, args.pool_size, args.max_waters,
                          args.sysdir, args.rep): s for s in args.systems}
        for fut in as_completed(futs):
            s, n, err = fut.result()
            print(f"  {s}: {n}/{args.n_frames} frames" +
                  (f" ({err})" if err else " OK"))


if __name__ == "__main__":
    main()
