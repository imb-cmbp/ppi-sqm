#!/usr/bin/env python3
"""Score one capped protein–protein interface with one semiempirical model.

The structure is a PDB with two partners: chain A is partner A and any
interfacial waters, chain B is partner B. The interaction energy is
E(AB) - E(A) - E(B) in kcal/mol, the same definition as in the paper.

Examples
--------
python scripts/score_model.py --list
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2-gbsa
python scripts/score_model.py --pdb model.pdb --model gfn2-alpb --charge-a 1 --charge-b -2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import score_xtb
import truncate_interface as TI
import xtb_runner as X
from cuby_jobs import fragment_nuclear_charges

# Friendly name -> key in score_xtb.VARIANTS. CPCM-X is omitted: the paper used
# the standalone CPCM-X program, not the --cpcmx flag built into xtb.
MODELS = {
    "gfn2-gbsa": "gfn2_gbsa",
    "gfn2-alpb": "gfn2_alpb",
    "gfn2-ddcosmo": "gfn2_cosmo",
    "gfn2-gas": "gfn2_gas",
    "gfn1-gbsa": "gfn1_gbsa",
    "gfn1-alpb": "gfn1_alpb",
    "gfn1-ddcosmo": "gfn1_cosmo",
}

LABELS = {
    "gfn2-gbsa": "GFN2-xTB/GBSA",
    "gfn2-alpb": "GFN2-xTB/ALPB",
    "gfn2-ddcosmo": "GFN2-xTB/ddCOSMO",
    "gfn2-gas": "GFN2-xTB, gas phase",
    "gfn1-gbsa": "GFN1-xTB/GBSA",
    "gfn1-alpb": "GFN1-xTB/ALPB",
    "gfn1-ddcosmo": "GFN1-xTB/ddCOSMO",
}


def list_models():
    width = max(len(name) for name in MODELS)
    for name in MODELS:
        print(f"{name:{width}}  {LABELS[name]}")


def _residues(pdb_path):
    groups = {}
    order = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        key = (line[21], line[22:26], line[26], resname)
        atom = line[12:16].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        el = (line[76:78].strip() or atom[:1]).upper()
        if key not in groups:
            groups[key] = {"resname": resname, "chain": line[21], "atoms": []}
            order.append(key)
        groups[key]["atoms"].append((atom, xyz, el))
    return [groups[key] for key in order]


def charges_from_pdb(pdb_path):
    """Formal partner charges, with the paper's closed-shell parity fix."""
    residues = _residues(pdb_path)
    chain_a = [r for r in residues if r["chain"] == "A"]
    chain_b = [r for r in residues if r["chain"] != "A"]
    if not chain_a or not chain_b:
        raise SystemExit(
            f"{pdb_path}: need chain A (partner A and waters) and chain B (partner B)")
    charge_a = TI._partner_charge(chain_a)
    charge_b = TI._partner_charge(chain_b)
    sum_z_a, sum_z_b = fragment_nuclear_charges(pdb_path)
    if sum_z_a is None:
        raise SystemExit(
            f"{pdb_path}: an element has no nuclear charge in the table; "
            "pass --charge-a and --charge-b")
    if (sum_z_a + sum_z_b - charge_a - charge_b) % 2 != 0:
        if charge_b <= charge_a:
            charge_b += 1
        else:
            charge_a += 1
    return charge_a, charge_b


def charges_from_sidecar(pdb_path):
    candidates = [
        Path(pdb_path).with_suffix(".trunc.json"),
        Path(pdb_path).with_name("capped.trunc.json"),
    ]
    for path in candidates:
        if path.is_file():
            meta = json.loads(path.read_text())
            return int(meta["charge_a"]), int(meta["charge_b"]), path
    return None


def resolve_xtb():
    """Point xtb_runner at a real executable without changing its default for other scripts."""
    if Path(X.XTB_EXE).is_file():
        return
    found = shutil.which("xtb")
    if found:
        X.XTB_EXE = Path(found)
        return
    raise SystemExit(
        "xtb was not found. Install xtb 6.7.1 and set XTB_HOME to the prefix "
        "that contains bin/xtb and share/xtb, or put xtb on PATH.")


def score(pdb_path, model, charge_a, charge_b, workdir, threads):
    variant = MODELS[model]
    gfn, solvent, solvation = score_xtb.VARIANTS[variant]
    workdir.mkdir(parents=True, exist_ok=True)
    energy, status = X.interaction_energy(
        pdb_path, workdir, charge_a, charge_b,
        solvent=solvent, gfn=gfn, threads=threads,
        solvation=solvation or "alpb")
    return energy, status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the model names and exit")
    ap.add_argument("--pdb", help="capped interface PDB (chain A, chain B)")
    ap.add_argument("--model", choices=sorted(MODELS), help="semiempirical model")
    ap.add_argument("--charge-a", type=int, default=None, help="formal charge of chain A")
    ap.add_argument("--charge-b", type=int, default=None, help="formal charge of chain B")
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads for xtb")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="directory for the xtb inputs and logs (default: ./score_<model>)")
    ap.add_argument("--charges-only", action="store_true",
                    help="print the partner charges and do not run xtb")
    args = ap.parse_args()

    if args.list:
        list_models()
        return
    if not args.pdb or not args.model:
        ap.error("give --pdb and --model, or --list")
    given = (args.charge_a is not None, args.charge_b is not None)
    if given == (True, False) or given == (False, True):
        ap.error("pass both --charge-a and --charge-b, or neither")

    pdb_path = Path(args.pdb)
    if not pdb_path.is_file():
        raise SystemExit(f"no such file: {pdb_path}")

    source = "command line"
    if given == (True, True):
        charge_a, charge_b = args.charge_a, args.charge_b
    else:
        sidecar = charges_from_sidecar(pdb_path)
        if sidecar is not None:
            charge_a, charge_b, side_path = sidecar
            source = str(side_path)
        else:
            charge_a, charge_b = charges_from_pdb(pdb_path)
            source = "residue names and chain termini in the PDB"

    print(f"model:    {LABELS[args.model]}")
    print(f"pdb:      {pdb_path}")
    print(f"charge_A: {charge_a}")
    print(f"charge_B: {charge_b}")
    print(f"charges:  {source}")
    if args.charges_only:
        return

    resolve_xtb()
    workdir = args.workdir or Path(f"score_{args.model}")
    energy, status = score(pdb_path, args.model, charge_a, charge_b, workdir, args.threads)
    print(f"status:   {status}")
    if energy is None:
        print(f"xtb logs: {workdir}")
        raise SystemExit(1)
    print(f"energy:   {energy:.4f} kcal/mol")
    print(f"xtb logs: {workdir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
