#!/usr/bin/env python3
"""Nwat-MM/GBSA following the protocol of Maffucci and Contini (2016).

Per system:
  * PyMOL InterfaceResidues (dSASA, cutoff 0.5) on complex_<ID>_leap.pdb gives the interface mask
  * for each Nwat in the scan: cpptraj `closest Nwat :interface` on prod4.mdcrd
    (center/image first), building complex/receptor trajectories + frame-1 PDBs
  * tleap-rebuild complex & receptor topologies (ff14SB/gaff/ions/extra-params,
    mbondi3) from the closest-water PDBs; ligand.top via parmstrip
  * MMPBSA.py (igb=8, saltcon=0.15), waters folded into the RECEPTOR
    (receptor_mask=recmask:lastres+1-lastres+Nwat), 100 frames (interval 10)
  * parse DELTA TOTAL

Writes a results CSV (system, nwat, dG, std) and prints the correlation with
experiment.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amber_setup import amber_bin, amber_env  # noqa
from paths import AMBERHOME, BENCHMARK, DATA, PYMOL_EXE, WORK  # noqa

REPO = WORK
ART = BENCHMARK
COMPLEXES = ART / "complexes"
EXTRA = ART / "protocol" / "extra-params"
INTERFACE_PML = ART / "protocol" / "interface.pymol"
SYS = REPO / "md" / "systems"
OUT = REPO / "md" / "mmgbsa"
RESULTS_CSV = REPO / "md" / "mmgbsa_results.csv"
PYMOL = PYMOL_EXE
FEATURES = ART / "protocol" / "complex_features.txt"


def load_features():
    feat = {}
    started = False
    for ln in FEATURES.read_text().splitlines():
        if ln.startswith("name"):
            started = True
            continue
        if not started or not ln.strip():
            continue
        p = ln.split()
        feat[p[0]] = {"rec": p[1], "lig": p[2], "last": int(p[3])}
    return feat


# POLAR setup: only these interface residues seed the water `closest` mask.
_POLAR = {"ARG", "ASH", "ASP", "GLH", "GLN", "GLU", "HID", "HIE", "HIP",
          "LYN", "LYS", "SER", "THR", "TRP", "TYR"}


def interface_mask(zid, wd, polar=False):
    """PyMOL InterfaceResidues (dASA cutoff 0.5) -> ':r1,r2,...'. With polar=True,
    keep only the polar interface residues (POLAR setup)."""
    pml = wd / "interface_run.pml"
    pml.write_text(
        f"run {INTERFACE_PML}\n"
        f"load {COMPLEXES}/complex_{zid}_leap.pdb\n"
        f'myInterfaceResidues = interfaceResidues("complex_{zid}_leap", '
        f'cA="c. A", cB="c. B", cutoff=0.5, selName="interface")\n'
        f"save {wd}/{zid}_interface.pdb, interface\nquit\n")
    subprocess.run([PYMOL, "-cq", str(pml)], cwd=wd, capture_output=True, text=True)
    ipdb = wd / f"{zid}_interface.pdb"
    if not ipdb.exists():
        return None
    resids = []
    for ln in ipdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            r = ln[22:26].strip()
            if polar and ln[17:20].strip() not in _POLAR:
                continue
            if r and r not in resids:
                resids.append(r)
    return ":" + ",".join(resids) if resids else None


def _cpptraj(inp, top, wd, env, tag):
    (wd / f"{tag}.cpptraj").write_text(inp)
    subprocess.run([amber_bin("cpptraj"), "-i", f"{tag}.cpptraj", "-p", str(top)],
                   cwd=wd, env=env, capture_output=True, text=True)


def _water_resrange(top_path, nwat, lastres):
    """Return (w_start, w_end) residue indices for the N explicit waters in a
    closest-stripped topology. After ``strip :Cl-,:Na+`` waters are contiguous at
    lastres+1..lastres+nwat in the complex top; rec.top must keep the same IDs."""
    script = (
        "import parmed as pmd\n"
        f"p = pmd.load_file({str(top_path)!r})\n"
        f"nw, lr = {nwat}, {lastres}\n"
        "wat = [r.idx + 1 for r in p.residues if r.name in ('WAT', 'HOH')]\n"
        "print(wat[0], wat[nw - 1]) if len(wat) >= nw else print(lr + 1, lr + nw)\n"
    )
    p = subprocess.run(["bash", "-c", f"source {AMBERHOME}/amber.sh && python3 -"],
                       input=script, capture_output=True, text=True, check=True)
    w, a = p.stdout.strip().split()
    return int(w), int(a)


def _fix_pdb_for_leap(pdb, out):
    """cpptraj PDBs often lack atom types; pdb4amber prepares them for tleap."""
    subprocess.run([amber_bin("pdb4amber"), "-i", str(pdb), "-o", str(out), "-y"],
                   capture_output=True, text=True)
    lines = [ln for ln in Path(out).read_text().splitlines()
             if not (ln.startswith(("ATOM", "HETATM")) and ln[12:16].strip() == "OXT")]
    Path(out).write_text("\n".join(lines) + "\n")


def _leap_rebuild(zid, nwat, wd, env):
    offs = "\n".join(f"loadoff {p}" for p in sorted(EXTRA.glob("*.off")))
    preps = "\n".join(f"loadamberprep {p}" for p in sorted(EXTRA.glob("*.prep")))
    frcs = "\n".join(f"loadamberparams {p}" for p in sorted(EXTRA.glob("*frcmod*")))
    (wd / "leap.in").write_text(
        "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\nsource leaprc.gaff\n"
        "loadamberparams frcmod.ionsjc_tip3p\n"
        "loadamberparams frcmod.ions1lm_1264_tip3p\n"
        "loadamberparams frcmod.ions234lm_1264_tip3p\n"
        f"{offs}\n{preps}\n{frcs}\n"
        "set default PBRadii mbondi3\n"
        f"cmp = loadpdb {zid}_cmp_fix.pdb\n"
        f"rec = loadpdb {zid}_rec_fix.pdb\n"
        f"saveamberparm cmp nwat{nwat}.{zid}.top nwat{nwat}.{zid}.crd\n"
        f"saveamberparm rec nwat{nwat}.{zid}.rec.top nwat{nwat}.{zid}.rec.crd\nquit\n")
    subprocess.run([amber_bin("tleap"), "-f", "leap.in"], cwd=wd, env=env,
                   capture_output=True, text=True)


def run_one(zid, nwat, intmask, feat, r=10, nprod=1, topology="cpptraj", force=False):
    # nprod>1: concatenate the last nprod production ns (prod{5-nprod..4}) for more
    # sampling. nprod=1 uses the fourth nanosecond only.
    wd = OUT / zid / (f"nwat{nwat}" if nprod == 1 else f"nwat{nwat}_p{nprod}")
    wd.mkdir(parents=True, exist_ok=True)
    fin = wd / f"FINAL_RESULTS_CLOSEST{nwat}"
    if not force and fin.exists():
        dg = parse_delta(fin.read_text())
        if dg is not None:
            return zid, nwat, dg
    env, _ = amber_env()
    top = SYS / zid / f"{zid}_complex_wat.top"
    prods = list(range(5 - nprod, 5))  # e.g. nprod=4 -> [1,2,3,4]
    trajf = [SYS / zid / f"{zid}_complex_wat.prod{p}.mdcrd" for p in prods]
    if not all(t.exists() for t in trajf):
        return zid, nwat, None
    traj = trajf[-1]  # kept for nwat=0 heredoc simplicity
    trajins = "\n".join(f"trajin {t} 1 1000 {r}" for t in trajf)
    rec, lig, last = feat["rec"], feat["lig"], feat["last"]
    nframes = nprod * (1000 // r)
    ctop = wd / f"nwat{nwat}.{zid}.top"
    rtop = wd / f"nwat{nwat}.{zid}.rec.top"
    ltop = wd / "ligand.top"
    mdc = wd / f"nwat{nwat}.{zid}.mdcrd"
    if nwat > 0:
        traj_src = trajf[-1]
        if topology == "tleap":
            # Trajectory, frame-1 PDBs and tleap topologies, as in the reference protocol.
            _cpptraj(f"{trajins}\ncenter @CA,C,N mass origin\n"
                     f"image origin center\nstrip :Cl-,:Na+\n"
                     f"closest {nwat} {intmask} noimage\n"
                     f"strip :OXT\ntrajout {mdc.name} nobox\n", top, wd, env, "cmplx_trj")
            pdb = wd / f"nwat{nwat}.{zid}.pdb"
            rpdb = wd / f"nwat{nwat}.{zid}.rec.pdb"
            _cpptraj(f"trajin {traj_src} 1 1\ncenter @CA,C,N mass origin\n"
                     f"image origin center\nstrip :Cl-,:Na+\n"
                     f"closest {nwat} {intmask} noimage\n"
                     f"strip :OXT\ntrajout {pdb.name} pdb\n", top, wd, env, "cmplx_pdb")
            _cpptraj(f"trajin {traj_src} 1 1\ncenter @CA,C,N mass origin\n"
                     f"image origin center\nstrip :Cl-,:Na+\n"
                     f"closest {nwat} {intmask} noimage\n"
                     f"strip {lig},:OXT\ntrajout {rpdb.name} pdb\n", top, wd, env, "rec_pdb")
            if not pdb.exists() or not rpdb.exists():
                return zid, nwat, None
            cmp_fix = wd / f"{zid}_cmp_fix.pdb"
            rec_fix = wd / f"{zid}_rec_fix.pdb"
            _fix_pdb_for_leap(pdb, cmp_fix)
            _fix_pdb_for_leap(rpdb, rec_fix)
            _leap_rebuild(zid, nwat, wd, env)
            if not ctop.exists() or not rtop.exists():
                return zid, nwat, None
            _cpptraj(f"parmstrip {rec},:Cl-,:Na+,:WAT\nparmbox nobox\n"
                     f"parmwrite out {ltop.name} amber\n", top, wd, env, "lig_gen")
            w, a = _water_resrange(ctop, nwat, last)
        else:
            # cpptraj closest-top split (fallback for nicked chains).
            for f in wd.glob("clo.*"):
                f.unlink(missing_ok=True)
            _cpptraj(f"{trajins}\ncenter @CA,C,N mass origin\n"
                     f"image origin center\nstrip :Cl-,:Na+\n"
                     f"closest {nwat} {intmask} noimage outprefix clo\n"
                     f"trajout {mdc.name} nobox\n", top, wd, env, "cmplx_trj")
            gen = list(wd.glob("clo.*"))
            if not gen:
                return zid, nwat, None
            _cpptraj(f"parmbox nobox\nparmwrite out {ctop.name} amber\n"
                     f"parmstrip {lig}\nparmwrite out {rtop.name} amber\n",
                     gen[0], wd, env, "topsplit")
            _cpptraj(f"parmbox nobox\nparmstrip {rec},:WAT\n"
                     f"parmwrite out {ltop.name} amber\n", gen[0], wd, env, "lig_gen")
            w, a = _water_resrange(ctop, nwat, last)
        recmask = f"{rec}:{w}-{a}"
    else:
        _cpptraj(f"{trajins}\nautoimage\nstrip :Cl-,:Na+,:WAT\n"
                 f"trajout {mdc.name} nobox\n", top, wd, env, "gen0")
        _cpptraj(f"parmstrip :Cl-,:Na+,:WAT\nparmbox nobox\n"
                 f"parmwrite out {ctop.name} amber\n"
                 f"parmstrip {lig}\nparmwrite out {rtop.name} amber\n",
                 top, wd, env, "topgen0")
        _cpptraj(f"parmstrip {rec},:Cl-,:Na+,:WAT\nparmbox nobox\n"
                 f"parmwrite out {ltop.name} amber\n", top, wd, env, "lig_gen")
        recmask = rec
    (wd / f"mmpbsa_closest_nwat{nwat}.in").write_text(
        f"&general\nreceptor_mask={recmask}, ligand_mask={lig}, "
        f"startframe=1, endframe={nframes}, interval=1, verbose=1,\n/\n"
        f"&gb\nigb=8, saltcon=0.15,\n/\n")
    # MMPBSA.py needs its Python modules (MMPBSA_mods) on PYTHONPATH, so source amber.sh
    mmcmd = (f"source {AMBERHOME}/amber.sh && MMPBSA.py -O "
             f"-i mmpbsa_closest_nwat{nwat}.in -o FINAL_RESULTS_CLOSEST{nwat} "
             f"-cp nwat{nwat}.{zid}.top -rp nwat{nwat}.{zid}.rec.top "
             f"-lp ligand.top -y nwat{nwat}.{zid}.mdcrd")
    subprocess.run(["bash", "-c", mmcmd], cwd=wd, env=env, capture_output=True, text=True)
    for junk in wd.glob("_MMPBSA_*"):
        try:
            junk.unlink()
        except OSError:
            pass
    if fin.exists():
        return zid, nwat, parse_delta(fin.read_text())
    return zid, nwat, None


def parse_delta(text):
    m = re.search(r"DELTA TOTAL\s+([-\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    global SYS, OUT, RESULTS_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--nwat", nargs="*", type=int, default=[0, 10, 20, 30, 40, 50])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nprod", type=int, default=1,
                    help="number of trailing production ns to concatenate (1=paper)")
    ap.add_argument("--sysdir", default=str(SYS))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--results-csv", default=str(RESULTS_CSV))
    ap.add_argument("--topology", choices=["tleap", "cpptraj"], default="cpptraj",
                    help="nwat>0 topology: cpptraj split (consistent traj/top) or tleap (PBS)")
    ap.add_argument("--force", action="store_true", help="recompute even if FINAL_RESULTS exist")
    ap.add_argument("--polar", action="store_true",
                    help="POLAR interface setup for the water mask")
    args = ap.parse_args()
    SYS = Path(args.sysdir)
    OUT = Path(args.outdir)
    RESULTS_CSV = Path(args.results_csv)
    feats = load_features()
    systems = args.systems or [s for s in feats if (SYS / s).exists() and s != "2B42"]
    OUT.mkdir(parents=True, exist_ok=True)
    # interface masks (serial: pymol)
    masks = {}
    for z in systems:
        wd = OUT / z
        wd.mkdir(parents=True, exist_ok=True)
        masks[z] = interface_mask(z, wd, polar=args.polar)
        print(f"{z}: interface mask {masks[z]}", flush=True)
    tasks = [(z, n, masks[z], feats[z], 10, args.nprod, args.topology, args.force)
             for z in systems if masks.get(z) for n in args.nwat]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, *t) for t in tasks]
        for fut in as_completed(futs):
            z, n, dg = fut.result()
            results[(z, n)] = dg
            print(f"{z} nwat{n}: dG={dg}", flush=True)
    csv = RESULTS_CSV
    with open(csv, "w") as f:
        f.write("System,Nwat,dG\n")
        for (z, n), dg in sorted(results.items()):
            f.write(f"{z},{n},{'' if dg is None else dg}\n")
    print(f"Wrote {csv}", flush=True)
    correlate(results, args.nwat, systems)


def correlate(results, nwats, systems):
    import csv as _csv
    from scipy import stats
    exp = {r["PDB_ID"]: float(r["Exp_dG"])
           for r in _csv.DictReader(open(DATA / "experimental_data.csv"))}
    print("\n=== Correlation vs experiment (per Nwat) ===")
    for n in nwats:
        xs, ys = [], []
        for z in systems:
            dg = results.get((z, n))
            if dg is not None and z in exp:
                xs.append(dg); ys.append(exp[z])
        if len(xs) >= 3:
            r = stats.pearsonr(xs, ys)[0]
            rho = stats.spearmanr(xs, ys)[0]
            print(f"  Nwat={n:2d}: N={len(xs):2d}  R2={r**2:.3f}  r={r:.3f}  rho={rho:.3f}")


if __name__ == "__main__":
    main()
