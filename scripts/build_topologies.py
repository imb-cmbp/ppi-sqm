#!/usr/bin/env python3
"""Build solvated Amber topologies with PROPKA (pH 7) protonation states.

Per system: propka3 on complex_<ID>_leap.pdb gives per-residue pKa values;
titratable states at pH 7 are assigned (HIP/HID/HIE, ASH/ASP, GLH/GLU, LYN/LYS,
CYM/CYS), keeping CYX, ACE/NME caps, Ca2, coordinates and numbering. Residues
whose state changes lose their H atoms, which tleap rebuilds (ff14SB/TIP3P/GAFF,
Joung-Cheatham and 12-6-4 ions, mbondi3, SS bonds, neutralization, truncated
octahedron with 10 A buffer), writing systems_propka/<ID>/<ID>_complex_wat.top/.crd.
A neutral His keeps the input HID/HIE tautomer; input HIP made neutral becomes HIE.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from paths import BENCHMARK, PROPKA_EXE, WORK

REPO = WORK
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amber_setup import amber_bin, amber_env  # noqa
import math


def find_disulfides(pdb):
    """Return list of (resi, resj) 1-based tleap residue indices for SG-SG < 2.5 A.
    tleap numbers residues sequentially across the loaded PDB (TER does not reset)."""
    sg = []
    ridx = 0
    cur = None
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21], line[22:27])
        if key != cur:
            cur = key
            ridx += 1
        if line[12:16].strip() == "SG":
            sg.append((ridx, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    bonds, used = [], set()
    for i in range(len(sg)):
        if sg[i][0] in used:
            continue
        for j in range(i + 1, len(sg)):
            if sg[j][0] in used:
                continue
            if math.dist(sg[i][1], sg[j][1]) < 2.5:
                bonds.append((sg[i][0], sg[j][0]))
                used.update({sg[i][0], sg[j][0]})
                break
    return bonds


# Amber22 cap atom names differ from those in the input PDBs.
_ACE_MAP = {"HH31": "H1", "HH32": "H2", "HH33": "H3"}
_NME_MAP = {"CH3": "C", "HH31": "H1", "HH32": "H2", "HH33": "H3"}

# titratable variant names mapped to standard names; propka does not titrate variants
_RESET = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP",
          "GLH": "GLU", "LYN": "LYS", "CYM": "CYS"}

ART = BENCHMARK
COMPLEXES = ART / "complexes"
EXTRA = ART / "protocol" / "extra-params"
OUT = REPO / "md" / "systems_propka"
PROPKA = PROPKA_EXE
BUFFER = 10.0
PH = 7.0

PDB_IDS = ["1ACB", "1ZHI", "1AVX", "2HLE", "1AY7", "2HRK", "1BVN", "2OOB", "1EMV",
           "2OUL", "1FLE", "2SIC", "1GLA", "2SNI", "1KAC", "2UUY", "1R0R", "3BZD",
           "1YVB", "3SGB"]

_METALS = {"Ca2"}


def run_propka(zid, work):
    """Return {(chain, resnum): (resn, pKa)} for titratable residues. Residue
    names are first reset to HIS/ASP/GLU/LYS/CYS, since propka does not titrate
    residues named HIP/HID/HIE/ASH/GLH/LYN."""
    src = work / f"{zid}_forpka.pdb"
    keep = []
    for ln in (COMPLEXES / f"complex_{zid}_leap.pdb").read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            rn = ln[17:20].strip()
            if rn in _RESET:
                ln = ln[:17] + f"{_RESET[rn]:>3}" + ln[20:]
            keep.append(ln)
        elif ln.startswith(("TER", "END")):
            keep.append(ln)
    src.write_text("\n".join(keep) + "\n")
    subprocess.run([PROPKA, "-o", str(PH), src.name], cwd=work,
                   capture_output=True, text=True)
    pka = work / f"{zid}_forpka.pka"
    out = {}
    if not pka.exists():
        return out
    insum = False
    for ln in pka.read_text().splitlines():
        if "SUMMARY OF THIS PREDICTION" in ln:
            insum = True
            continue
        if insum:
            m = re.match(r"\s*([A-Z0-9]+)\s+(\d+)\s+([A-Za-z0-9])\s+([\d.]+)\s+([\d.]+)", ln)
            if m:
                out[(m.group(3), m.group(2))] = (m.group(1), float(m.group(4)))
            elif "Free energy" in ln or ln.strip().startswith("-"):
                break
    return out


def moe_hisname(zid):
    """His tautomer per residue in the input PDB, inferred from its H atoms."""
    res = {}
    for ln in (COMPLEXES / f"complex_{zid}_leap.pdb").read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            key = (ln[21], ln[22:26].strip())
            res.setdefault(key, [ln[17:20].strip(), set()])[1].add(ln[12:16].strip())
    out = {}
    for k, (rn, names) in res.items():
        if rn in ("HID", "HIE", "HIP", "HIS"):
            hd1, he2 = "HD1" in names, "HE2" in names
            out[k] = "HIP" if (hd1 and he2) else ("HID" if hd1 else "HIE" if he2 else "HIE")
    return out


def decide(resn, pka):
    if resn == "HIS":
        return "HIP" if pka > PH else "NEUTRAL"   # tautomer chosen by caller
    if resn == "ASP":
        return "ASH" if pka > PH else "ASP"
    if resn == "GLU":
        return "GLH" if pka > PH else "GLU"
    if resn == "LYS":
        return "LYS" if pka > PH else "LYN"
    if resn == "CYS":
        return "CYM" if pka < PH else "CYS"
    return None


def reprotonate_pdb(zid, work):
    """Write the input PDB with PROPKA states applied; return (path, state counts)."""
    pk = run_propka(zid, work)
    moehis = moe_hisname(zid)
    lines = (COMPLEXES / f"complex_{zid}_leap.pdb").read_text().splitlines()
    newname = {}   # (chain,num) -> new resname
    summary = {"HIP->neutral": 0, "ASH": 0, "GLH": 0, "LYN": 0, "CYM": 0, "HIP": 0}
    for key, (resn, pka) in pk.items():
        d = decide(resn, pka)
        if d is None:
            continue
        if resn == "HIS":
            if d == "HIP":
                newname[key] = "HIP"
                summary["HIP"] += 1
            else:
                # neutral: keep the input tautomer if it was neutral, else HIE
                mt = moehis.get(key, "HIE")
                newname[key] = "HID" if mt == "HID" else "HIE"
                if moehis.get(key) == "HIP":
                    summary["HIP->neutral"] += 1
        else:
            newname[key] = d
            if d in ("ASH", "GLH", "LYN", "CYM"):
                summary[d] += 1
    # Keep all atoms and rename cap atoms. Residues whose protonation state
    # changes lose their H atoms and are renamed so tleap rebuilds them; all
    # other residues (including caps, CYX and metals) are kept unchanged.
    out = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            resn = ln[17:20].strip()
            atom = ln[12:16].strip()
            key = (ln[21], ln[22:26].strip())
            # cap atom renaming (Amber22 names)
            mp = _ACE_MAP if resn == "ACE" else _NME_MAP if resn == "NME" else None
            if mp and atom in mp:
                ln = ln[:12] + f"{mp[atom]:<4}"[:4] + ln[16:]
            if key in newname and newname[key] != resn:
                # state changed: strip this residue's H, rename; tleap re-adds H
                elem = ln[76:78].strip() if len(ln) >= 78 else ""
                if elem == "H" or atom.startswith("H") or re.match(r"\d+H", atom):
                    continue
                ln = ln[:17] + f"{newname[key]:>3}" + ln[20:]
            out.append(ln)
        elif ln.startswith(("TER", "END")):
            out.append(ln)
    hv = work / f"{zid}_propka_heavy.pdb"
    hv.write_text("\n".join(out) + "\n")
    return hv, summary


def build(zid, tleap):
    work = OUT / zid
    work.mkdir(parents=True, exist_ok=True)
    pdb, summary = reprotonate_pdb(zid, work)
    bonds = find_disulfides(pdb)
    ss = "\n".join(f"bond x.{i}.SG x.{j}.SG" for i, j in bonds)
    offs = "\n".join(f"loadoff {p}" for p in sorted(EXTRA.glob("*.off")))
    preps = "\n".join(f"loadamberprep {p}" for p in sorted(EXTRA.glob("*.prep")))
    frcs = "\n".join(f"loadamberparams {p}" for p in sorted(EXTRA.glob("*frcmod*")))
    leapin = work / "build.leap"
    leapin.write_text(
        "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\nsource leaprc.gaff\n"
        "loadamberparams frcmod.ionsjc_tip3p\n"
        "loadamberparams frcmod.ions1lm_1264_tip3p\n"
        "loadamberparams frcmod.ions234lm_1264_tip3p\n"
        f"{offs}\n{preps}\n{frcs}\n"
        "set default PBRadii mbondi3\n"
        f"x = loadpdb {pdb.name}\n{ss}\n"
        "addions x Na+ 0\naddions x Cl- 0\n"
        f"solvateoct x TIP3PBOX {BUFFER}\n"
        f"saveamberparm x {zid}_complex_wat.top {zid}_complex_wat.crd\n"
        f"savepdb x {zid}_solv.pdb\nquit\n")
    os.system(f"cd {work} && {tleap} -f {leapin.name} > build.leap.out 2>&1")
    top = work / f"{zid}_complex_wat.top"
    ok = top.exists() and top.stat().st_size > 0
    s = ", ".join(f"{k}={v}" for k, v in summary.items() if v)
    return f"{zid}: {'OK' if ok else 'FAILED'}  ss={len(bonds)}  [{s}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    args = ap.parse_args()
    tleap = amber_bin("tleap")
    OUT.mkdir(parents=True, exist_ok=True)
    for z in args.systems:
        print(build(z, tleap), flush=True)


if __name__ == "__main__":
    main()
