#!/usr/bin/env python3
"""Score one capped protein–protein interface with one semiempirical model.

The structure is a PDB with two partners: chain A is partner A and any
interfacial waters, chain B is partner B. The interaction energy is
E(AB) - E(A) - E(B) in kcal/mol, the same definition as in the paper.
GFN-xTB models run xtb directly; GFN2-xTB/CPCM-X runs xtb with the standalone
cpx program; PM6, PM7 and PM6-ML models run MOPAC through cuby4.

Examples
--------
python scripts/score_model.py --list
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2_gbsa
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model pm6_d3h4x_cosmo2
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2_cpcmx --threads 4
python scripts/score_model.py --pdb model.pdb --model gfn2_gbsa --max-waters 20 --water-assign nearer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cuby_jobs as QM
import score_mopac as S
import score_xtb
import truncate_interface as TI
import xtb_runner as X
from common import parse_interaction_energy, safe_run
from interaction import SQM_PRESETS
from paths import CPCMX_DB, CPX_EXE
from protocol_names import protocol_display_name

# Energy models scored in the paper. gfn2_cpcmx uses the standalone CPCM-X
# program (workflow/03_cpcmx.sh); the others are workflow/02_score.sh.
XTB_MODELS = ["gfn2_gbsa", "gfn2_alpb", "gfn2_cosmo", "gfn1_gbsa", "gfn1_alpb",
              "gfn1_cosmo", "gfn2_cpcmx"]
MOPAC_MODELS = [
    "pm6_d3h4_cosmo_nofc", "pm6_d3h4_cosmo_cuby", "pm6_d3h4_cosmo", "pm6_d3h4s_cosmo",
    "pm6_d3h4s_cosmo2", "pm6_d3h4x_cosmo2", "pm6_d3h4paren_cosmo", "pm6_d3_cosmo",
    "pm6_h4_cosmo_cuby", "pm6_dh2_cosmo", "pm6_dhplus_cosmo", "pm6_dhplus_cosmo2",
    "pm6_cosmo", "pm6_cosmo2", "pm6_d3h4x", "pm7", "pm7_cosmo2",
    "pm6_ml", "pm6_ml_cosmo", "pm6_ml_cosmo2",
]
MODELS = (
    {name: "xtb" for name in XTB_MODELS if name != "gfn2_cpcmx"}
    | {"gfn2_cpcmx": "CPCM-X"}
    | {name: "MOPAC" for name in MOPAC_MODELS}
)

# Written into every work directory so that a rerun only ever deletes its own files.
MARKER = ".score_model"


def label(name):
    return protocol_display_name(name.upper(), omit_even10=True)


def list_models():
    width = max(len(name) for name in MODELS)
    for name, program in MODELS.items():
        print(f"{name:{width}}  {program:6}  {label(name)}")


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
    sum_z_a, sum_z_b = QM.fragment_nuclear_charges(pdb_path)
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


def _found(exe):
    return bool(exe) and (Path(exe).is_file() or shutil.which(str(exe)) is not None)


def check_programs(model):
    """Stop with an installation hint when a program the model needs is missing."""
    missing = []
    kind = MODELS[model]
    if kind in ("xtb", "CPCM-X"):
        if not Path(X.XTB_EXE).is_file():
            missing.append("xtb 6.7.1: install it (it is in environment.yml) or set "
                           "XTB_HOME to the prefix that contains bin/xtb")
    if kind == "CPCM-X":
        if not _found(CPX_EXE):
            missing.append("cpx (CPCM-X 1.1.0): install it (it is in environment.yml) "
                           "or set CPX_EXE")
        if not (Path(CPCMX_DB) / "crs.param_h2o").is_file():
            missing.append(
                f"CPCM-X parameter files: clone https://github.com/grimme-lab/CPCM-X "
                f"and set CPCMX_DB to its DB/xtb directory (tried {CPCMX_DB})")
    if kind == "MOPAC":
        if not (Path(S.CUBY4_DIR, "cuby4").is_file() or _found("cuby4")):
            missing.append("cuby4: download it from http://cuby4.molecular.cz and set "
                           "CUBY4_DIR to the directory that contains the cuby4 script")
        if not ((S.RUBY_BIN and Path(S.RUBY_BIN, "ruby").is_file()) or _found("ruby")):
            missing.append("Ruby, which runs cuby4: put ruby on PATH or set RUBY_BIN "
                           "to its bin directory")
        exe = QM.resolve_mopac_exe(model, S.MOPAC_PATH)
        if not _found(exe):
            var = "MOPAC_ML_EXE" if SQM_PRESETS[model].get("is_pm6_ml") else "MOPAC_EXE"
            missing.append(f"{exe}: set {var} to the executable")
        elif Path(exe).name == "mopac-ml-cuby":
            ml_dir = os.environ.get("MOPAC_ML_DIR", "")
            ml_py = os.environ.get("PM6ML_PYTHON", "")
            if not Path(ml_dir, "mopac-ml").is_file():
                missing.append("MOPAC-ML: clone https://github.com/Honza-R/mopac-ml and set "
                               "MOPAC_ML_DIR to that directory")
            if not _found(ml_py):
                missing.append("PM6-ML Python: set PM6ML_PYTHON to the python of the "
                               "environment with torchmd-net, simple-dftd3 and dftd3-python")
    if missing:
        raise SystemExit(f"Cannot run {model}:\n  " + "\n  ".join(missing))


def prepare_workdir(workdir):
    if workdir.exists() and any(workdir.iterdir()) and not (workdir / MARKER).exists():
        raise SystemExit(f"{workdir} exists and was not created by score_model.py; "
                         "pass another --workdir")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / MARKER).touch()


def prepare_geometry(pdb_path, workdir, max_waters, water_assign):
    """Copy the PDB into the work directory, applying water options.

    Returns (geometry path, notes). Charges are read from the original PDB or
    its sidecar before this runs, so trimming or reassigning waters does not
    change them (water is neutral).
    """
    geom = workdir / "geom.pdb"
    shutil.copy(pdb_path, geom)
    notes = []
    if max_waters is not None:
        n = TI.trim_waters(geom, geom, max_waters)
        notes.append(f"kept {n} waters (--max-waters {max_waters})")
    if water_assign == "nearer":
        info = TI.assign_waters_nearer(geom, geom)
        notes.append(
            f"water-assign nearer: {info['n_waters_a']} with A, "
            f"{info['n_waters_b']} with B")
    return geom, notes


def score_xtb_model(pdb_path, model, charge_a, charge_b, workdir, threads):
    gfn, solvent, solvation = score_xtb.VARIANTS[model]
    energy, status = X.interaction_energy(
        pdb_path, workdir, charge_a, charge_b,
        solvent=solvent, gfn=gfn, threads=threads, solvation=solvation or "alpb")
    return energy, status


def score_cpcmx_model(pdb_path, charge_a, charge_b, workdir, threads):
    """GFN2-xTB/CPCM-X as in workflow/03_cpcmx.sh (score_cpcmx.score_frame)."""
    import score_cpcmx as CX
    keep = workdir / "logs"
    keep.mkdir(exist_ok=True)
    row, status, _rss = CX.score_frame(
        pdb_path, workdir / "cpcmx", charge_a, charge_b,
        threads, "512M", 7200, keep)
    if row is None:
        return None, status
    (workdir / "cpcmx.json").write_text(json.dumps(row, indent=1))
    return row["energy"], status


def score_mopac_model(pdb_path, model, charge_a, charge_b, workdir, threads,
                      water_assign="receptor"):
    """cuby4 interaction job with the charge search of score_mopac.py.

    MOZYME may reject a formal charge in favour of the charge of its own Lewis
    structure; the charges it reports are used for the next attempt.
    """
    geom = workdir / "geom.pdb"
    if Path(pdb_path).resolve() != geom.resolve():
        shutil.copy(pdb_path, geom)
    # With nearer assignment the chain letters already encode the split, so
    # cuby's receptor mode reads them correctly. Explicit 'partition' is the
    # same geometric rule applied at YAML time without rewriting the PDB.
    assign = "receptor"
    sum_z_a, sum_z_b = QM.fragment_nuclear_charges(geom, assign)
    sum_z = None if sum_z_a is None else sum_z_a + sum_z_b
    env = S.sqm_thread_env(threads)
    yaml_f = workdir / "interaction.yaml"
    known, tried = {}, set()
    status = "no charge pair was accepted by MOPAC"
    while True:
        guess = QM.next_charge_guess(charge_a, charge_b, sum_z, known, tried,
                                     sum_z_a, sum_z_b)
        if guess is None:
            return None, status
        if tried:
            S._clean_workdir(workdir, keep=("geom.pdb", MARKER))
        tried.add(guess)
        QM.write_interaction_yaml(
            yaml_f, geom, model, S.MOPAC_PATH, threads, *guess,
            water_assign=assign)
        out, err, code = safe_run(f"cuby4 {yaml_f.name}", workdir, env)
        (workdir / "cuby4.log").write_text(out + err)
        if code == 0:
            energy = parse_interaction_energy(out)
            if math.isnan(energy):
                return None, "cuby4 printed no interaction energy"
            if guess == (charge_a, charge_b):
                return energy, "ok"
            return energy, f"ok with charges A={guess[0]}, B={guess[1]}"
        status = f"cuby4 exited with code {code}"
        if not S._sqm_retryable(err):
            return None, status
        known.update(QM.mozyme_computed_charges(workdir))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the model names and exit")
    ap.add_argument("--pdb", help="capped interface PDB (chain A, chain B)")
    ap.add_argument("--model", choices=list(MODELS), metavar="MODEL",
                    help="energy model, see --list")
    ap.add_argument("--charge-a", type=int, default=None, help="formal charge of chain A")
    ap.add_argument("--charge-b", type=int, default=None, help="formal charge of chain B")
    ap.add_argument("--max-waters", type=int, default=None,
                    help="keep only the first N waters of the PDB (file order; "
                         "the paper's models are ranked by bridging score)")
    ap.add_argument("--water-assign", choices=["receptor", "nearer"], default="receptor",
                    help="receptor (default): all waters with partner A, the "
                         "Nwat-MM/GBSA convention used in the paper; nearer: each "
                         "water with the closer partner (Supporting Information S7)")
    ap.add_argument("--threads", type=int, default=1, help="threads for xtb, cpx or MOPAC")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="directory for inputs and logs (default: ./score_<pdb>_<model>)")
    ap.add_argument("--charges-only", action="store_true",
                    help="print the partner charges and do not run the calculation")
    args = ap.parse_args()

    if args.list:
        list_models()
        return
    if not args.pdb or not args.model:
        ap.error("give --pdb and --model, or --list")
    given = (args.charge_a is not None, args.charge_b is not None)
    if given == (True, False) or given == (False, True):
        ap.error("pass both --charge-a and --charge-b, or neither")

    pdb_path = Path(args.pdb).resolve()
    if not pdb_path.is_file():
        raise SystemExit(f"no such file: {args.pdb}")

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

    print(f"model:    {args.model} ({label(args.model)})")
    print(f"pdb:      {args.pdb}")
    print(f"charge_A: {charge_a}")
    print(f"charge_B: {charge_b}")
    print(f"charges:  {source}")
    print(f"waters:   assign={args.water_assign}"
          + (f", max={args.max_waters}" if args.max_waters is not None else ""))
    if args.charges_only:
        return

    check_programs(args.model)
    workdir = (args.workdir or Path(f"score_{pdb_path.stem}_{args.model}")).resolve()
    prepare_workdir(workdir)
    geom, notes = prepare_geometry(pdb_path, workdir, args.max_waters, args.water_assign)
    for note in notes:
        print(f"note:     {note}")

    kind = MODELS[args.model]
    if kind == "xtb":
        energy, status = score_xtb_model(
            geom, args.model, charge_a, charge_b, workdir, args.threads)
    elif kind == "CPCM-X":
        energy, status = score_cpcmx_model(
            geom, charge_a, charge_b, workdir, args.threads)
    else:
        energy, status = score_mopac_model(
            geom, args.model, charge_a, charge_b, workdir, args.threads,
            water_assign=args.water_assign)
    print(f"status:   {status}")
    print(f"logs:     {workdir}")
    if energy is None:
        raise SystemExit(1)
    print(f"energy:   {energy:.4f} kcal/mol")


if __name__ == "__main__":
    sys.exit(main() or 0)
