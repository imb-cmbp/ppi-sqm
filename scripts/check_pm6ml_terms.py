#!/usr/bin/env python3
"""Recompute the D3(BJ) and machine-learned terms of PM6-ML for capped interface
models, as added by the MOPAC-ML wrapper, so the PM6-ML minus PM6 interaction
energy in the Cuby logs can be checked term by term. Requires torchmd-net and dftd3.
Usage: python3 check_pm6ml_terms.py path/to/capped.pdb ...
"""
import os, sys, numpy as np, torch
torch.set_num_threads(4)
from torchmdnet.models.model import load_model
from dftd3.interface import RationalDampingParam, DispersionModel
MODEL = os.environ.get("PM6ML_MODEL", "PM6-ML_correction_seed8_best.ckpt")
HARTREE2KJMOL = 627.5094740631 * 4.184
BOHR = 0.529177210903
Z = {"H":1,"C":6,"N":7,"O":8,"S":16,"P":15,"F":9,"Cl":17,"Br":35,"I":53,"Na":11,"Mg":12,"K":19,"Ca":20,"Li":3}
AT = {35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13, 3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26}
model = load_model(MODEL, derivative=True)
def elem(line):
    e = line[76:78].strip()
    if not e:
        e = "".join(c for c in line[12:16] if c.isalpha())[:1]
    return e.capitalize()
def read(pdb):
    atoms = []
    for l in open(pdb):
        if l.startswith(("ATOM","HETATM")):
            atoms.append((l[21], Z[elem(l)], float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return atoms
def terms(sel):
    z = np.array([a[1] for a in sel]); xyz = np.array([a[2:] for a in sel])
    d3 = DispersionModel(numbers=z, positions=xyz / BOHR).get_dispersion(
        RationalDampingParam(s6=1.0, s8=0.3908, a1=0.566, a2=3.128), grad=False)["energy"] * HARTREE2KJMOL
    e, _ = model.forward(torch.tensor([AT[i] for i in z]), torch.tensor(xyz, dtype=torch.float32))
    return d3 / 4.184, e.item() / 4.184
for pdb in sys.argv[1:]:
    atoms = read(pdb)
    first = sorted({a[0] for a in atoms})[0]
    A = [a for a in atoms if a[0] == first]; B = [a for a in atoms if a[0] != first]
    dab, mab = terms(atoms); da, ma = terms(A); db, mb = terms(B)
    print(f"{pdb}\n  dD3 {dab-da-db:9.2f}  dML {mab-ma-mb:9.2f}  sum {dab-da-db+mab-ma-mb:9.2f} kcal/mol", flush=True)
