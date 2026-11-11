#!/usr/bin/env python3
"""Extract 10 scoring snapshots per system from the prod4 trajectory.

Frames are taken from the 1000-frame fourth production nanosecond with stride 100,
the same sampling as the MM-GBSA calculation. Dry PDBs have solvent and ions
stripped; wet PDBs keep the Nwat=30 waters closest to the polar dASA (0.5)
interface mask (cpptraj ``closest 30 :polar_mask``).
Writes <out-base>/<SYS>/ff14sb/<rep>/{dry,wet}_f<i>.pdb.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHAIN_REF, WORK  # noqa: E402

REPO = WORK

from amber_setup import amber_bin, amber_env  # noqa
from common import safe_run, get_first_chain_nres, get_protein_residue_count  # noqa
from common import sanitize_wet_pdb_to_hoh, PDB_IDS  # noqa
import interface_dasa as ID  # noqa

SYSDIR = REPO / "md" / "systems_propka_r1"
OUT_BASE = REPO / "screening_ff_md_sqm_propka_r1_w30"
PDB_INPUT = CHAIN_REF
N_FRAMES = 10
STRIDE = 100          # frames 1, 101, ..., 901 of prod4 (1000 frames)
MAX_WATERS = 30       # Nwat


def _cpptraj(inp, top, wd, env, tag):
    (wd / f"{tag}.cpptraj").write_text(inp)
    safe_run(f"{amber_bin('cpptraj')} -i {tag}.cpptraj -p {top}", wd, env)


def extract_one(system, out_base, n_frames, stride, max_waters, polar_dasa,
                sysdir_root=None, rep="rep3"):
    env, _ = amber_env()
    sysdir = Path(sysdir_root or SYSDIR).resolve() / system
    top = (sysdir / f"{system}_complex_wat.top").resolve()
    traj = (sysdir / f"{system}_complex_wat.prod4.mdcrd").resolve()
    if not top.exists() or not traj.exists():
        return system, 0, "missing top/traj"

    out_dir = Path(out_base).resolve() / system / "ff14sb" / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_pdb = PDB_INPUT / f"{system}_input.pdb"
    n_chain_a = get_first_chain_nres(inp_pdb)
    prot_res = get_protein_residue_count(inp_pdb)

    mask = ID.polar_mask_for_waters(system) if polar_dasa else None
    if polar_dasa and not mask:
        return system, 0, "no polar dASA mask"

    done = 0
    for i in range(1, n_frames + 1):
        fr = 1 + (i - 1) * stride
        dry_dst = out_dir / f"dry_f{i}.pdb"
        wet_dst = out_dir / f"wet_f{i}.pdb"
        if dry_dst.exists() and wet_dst.exists():
            done += 1
            continue

        work = out_dir / f"_extract_f{i}"
        work.mkdir(exist_ok=True)

        _cpptraj(
            f"trajin {traj} {fr} {fr}\n"
            "autoimage\n"
            "strip :WAT,Cl-,Na+\n"
            f"trajout {dry_dst.name}\nrun\nquit\n",
            top, work, env, "dry")

        if dry_dst.exists() or (work / dry_dst.name).exists():
            src = work / dry_dst.name if (work / dry_dst.name).exists() else dry_dst
            if src != dry_dst:
                src.replace(dry_dst)

        if mask:
            for f in work.glob("clo.*"):
                f.unlink(missing_ok=True)
            _cpptraj(
                f"trajin {traj} {fr} {fr}\n"
                "center @CA,C,N mass origin\n"
                "image origin center\n"
                "strip :Na+,Cl-\n"
                f"closest {max_waters} {mask} noimage outprefix clo\n"
                f"trajout {wet_dst.name} nobox\nrun\nquit\n",
                top, work, env, "wet")
            gen = list(work.glob("clo.*"))
            if gen:
                _cpptraj(
                    f"trajin {gen[0].name} 1 1\n"
                    f"trajout {wet_dst.name}\nrun\nquit\n",
                    top, work, env, "wet2")
        else:
            _cpptraj(
                f"trajin {traj} {fr} {fr}\n"
                "autoimage\n"
                f"trajout {wet_dst.name}\nrun\nquit\n",
                top, work, env, "wet")

        w = work / wet_dst.name
        if w.exists():
            w.replace(wet_dst)
        if wet_dst.exists():
            sanitize_wet_pdb_to_hoh(wet_dst)

        if dry_dst.exists() and wet_dst.exists():
            done += 1

        for junk in work.glob("*"):
            junk.unlink(missing_ok=True)
        work.rmdir()

    return system, done, None if done == n_frames else f"only {done}/{n_frames}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--out-base", default=str(OUT_BASE))
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--max-waters", type=int, default=MAX_WATERS)
    ap.add_argument("--no-polar-dasa", action="store_true",
                    help="skip POLAR dASA mask for wet waters")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sysdir", default=str(SYSDIR),
                    help="md/systems_* directory with prod4 trajectories")
    ap.add_argument("--rep", default="rep3",
                    help="replica folder name written under <sys>/ff14sb/")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    polar = not args.no_polar_dasa
    print(f"Precomputing dASA masks (polar={polar})...")
    for s in args.systems:
        ID.get_mask(s, dasa_cutoff=0.5, polar=False)
        if polar:
            ID.get_mask(s, dasa_cutoff=0.5, polar=True)

    tasks = args.systems
    print(f"Extracting {args.n_frames} frames (stride {args.stride}) for {len(tasks)} systems "
          f"from {args.sysdir} -> {out_base} /ff14sb/{args.rep}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, s, out_base, args.n_frames, args.stride,
                          args.max_waters, polar, args.sysdir, args.rep): s for s in tasks}
        for fut in as_completed(futs):
            s, n, err = fut.result()
            print(f"  {s}: {n}/{args.n_frames} frames" + (f" ({err})" if err else " OK"))


if __name__ == "__main__":
    main()
