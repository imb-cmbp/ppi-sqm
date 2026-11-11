#!/usr/bin/env python3
"""Per-complex structural statistics, written to structure_statistics.csv.

Residue and atom counts for each partner, residues, waters and atoms of the capped
interface model, mean basin count per clustering metric, and backbone RMSD
summaries from rmsd_analysis.csv. Reads existing structure and clustering output.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from paths import BENCHMARK, DATA, WORK

REPO = WORK
ART = REPO / "md"
SCR = REPO / "screening_ff_md_sqm"
REPS = ["r1", "r2", "r3", "r4", "r5"]
OUT = DATA / "structure_statistics.csv"

COMPLEXES = BENCHMARK / "complexes"

exp = pd.read_csv(DATA / "experimental_data.csv")
exp.columns = ["System", "Exp_dG"] + list(exp.columns[2:])
expd = dict(zip(exp.System, exp.Exp_dG))

rows = []
for system in sorted(expd):
    rec = {"System": system, "Exp_dG": expd[system]}

    # intact complex: residues and atoms per chain
    pdb = COMPLEXES / f"complex_{system}_leap.pdb"
    if pdb.exists():
        ch = defaultdict(set)
        at = defaultdict(int)
        for ln in pdb.read_text(errors="ignore").splitlines():
            if ln.startswith("ATOM"):
                c = ln[21]
                ch[c].add(ln[22:27])
                at[c] += 1
        keys = sorted(ch)
        if len(keys) >= 2:
            rec["res_A"] = len(ch[keys[0]])
            rec["res_B"] = len(ch[keys[1]])
            rec["atoms_complex"] = sum(at.values())

    # truncated model: atoms, retained residues and waters
    cap = None
    for f in sorted((SCR / "_qmmm_trunc_propka" / "r1" / system).rglob("capped.pdb")):
        cap = f
        break
    if cap:
        lines = [l for l in cap.read_text(errors="ignore").splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
        rec["atoms_model"] = len(lines)
        res = {(l[21], l[22:27]) for l in lines if l[17:20] not in ("HOH", "WAT")}
        rec["res_model"] = len(res)
        rec["waters_model"] = len({l[22:27] for l in lines if l[17:20] in ("HOH", "WAT")})

    # basins found by each clustering metric, averaged over trajectories
    for tag in ("aa", "bb", "iface_sc", "contacts"):
        base = REPO / f"screening_ff_md_sqm_propka_clust_{tag}_gt5"
        if not base.exists():
            base = REPO / f"screening_ff_md_sqm_propka_clust_{tag}"
        nb, cov = [], []
        for rep in REPS:
            bf = base / system / "ff14sb" / rep / "basins.json"
            if bf.exists():
                d = json.loads(bf.read_text())
                nb.append(d.get("n_basins", np.nan))
                cov.append(d.get("covered_frac", np.nan))
        if nb:
            rec[f"basins_{tag}"] = round(float(np.nanmean(nb)), 1)
            if tag == "aa":
                rec["coverage_aa"] = round(100 * float(np.nanmean(cov)), 1)
    rows.append(rec)

df = pd.DataFrame(rows)

# backbone RMSD summary per complex, across the five trajectories
rf = DATA / "rmsd_analysis.csv"
if rf.exists():
    r = pd.read_csv(rf)
    g = r.groupby("system").rmsd_mean
    df["rmsd_mean"] = df.System.map(g.mean().round(2))
    df["rmsd_min"] = df.System.map(g.min().round(2))
    df["rmsd_max"] = df.System.map(g.max().round(2))

df = df.sort_values("Exp_dG")
df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(df)} complexes)")
print(df.to_string(index=False))
