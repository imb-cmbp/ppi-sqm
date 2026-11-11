"""Small helpers shared by the preparation, extraction and scoring scripts."""
from __future__ import annotations

import os
import subprocess

import numpy as np

PDB_IDS = [
    "1ACB", "1ZHI", "1AVX", "2HLE", "1AY7", "2HRK", "1BVN", "2OOB", "1EMV", "2OUL",
    "1FLE", "2SIC", "1GLA", "2SNI", "1KAC", "2UUY", "1R0R", "3BZD", "1YVB", "3SGB",
]

WATER_RESNAMES = {"WAT", "HOH", "OPC", "SOL", "TIP3", "TIP4", "T4D", "TP4"}


def sanitize_wet_pdb_to_hoh(wet_pdb):
    """Keep only O/H atoms for water residues and normalize water name to HOH."""
    if not wet_pdb.exists():
        return

    kept = []
    for line in wet_pdb.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            kept.append(line)
            continue

        resname = line[17:20].strip().upper()
        atom_name = line[12:16].strip().upper()
        if resname in WATER_RESNAMES:
            if atom_name.startswith(("O", "H")):
                kept.append(f"{line[:17]}HOH{line[20:]}")
            continue
        kept.append(line)

    wet_pdb.write_text("\n".join(kept) + "\n")


def get_protein_residue_count(complex_pdb):
    """Highest residue number in the file."""
    max_res = 0
    with open(complex_pdb, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    res_num = int(line[22:26].strip())
                    max_res = max(max_res, res_num)
                except ValueError:
                    pass
    return max_res


def get_first_chain_nres(complex_pdb):
    """Number of residues in the first chain of the file."""
    first_chain = None
    nres = 0
    last_key = None
    with open(complex_pdb, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                if first_chain is None:
                    first_chain = chain
                if chain != first_chain:
                    break
                key = line[22:27]
                if key != last_key:
                    nres += 1
                    last_key = key
    return nres


def safe_run(cmd, cwd, env_vars=None, err_file=None):
    """Run a shell command; stderr goes to err_file (inside cwd) when given."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    if err_file:
        with open(cwd / err_file, "w") as ef:
            proc = subprocess.Popen(
                cmd, cwd=cwd, shell=True, env=env,
                stdout=subprocess.PIPE, stderr=ef, text=True,
            )
            stdout, _ = proc.communicate()
            return stdout, "", proc.returncode
    else:
        proc = subprocess.Popen(
            cmd, cwd=cwd, shell=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = proc.communicate()
        return stdout, stderr, proc.returncode


def parse_interaction_energy(stdout):
    """Interaction energy (kcal/mol) from cuby4 output, NaN if absent."""
    for line in stdout.splitlines():
        if "Interaction energy:" in line:
            try:
                return float(line.split()[-2])
            except ValueError:
                pass
    return np.nan
