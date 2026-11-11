#!/usr/bin/env python3
"""GFN-xTB single points and interaction energies on capped interface models.

Runs the ``xtb`` executable directly, following the MOPAC protocol of cuby_jobs.py:
three single points (AB, A, B) on the same geometry, with per-partner charges from
the truncation sidecar. Chain A is partner A plus interfacial waters, chain B is
partner B. Also parses the implicit-solvation terms printed by xtb.
"""
from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
from pathlib import Path

from paths import XTB_HOME
XTB_EXE = XTB_HOME / "bin" / "xtb"
HARTREE_KCAL = 627.5094740631

_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_ALPB_RES = {
    "Gsolv": re.compile(r"-> Gsolv\s+(-?\d+\.\d+)\s+Eh"),
    "Gelec": re.compile(r"-> Gelec\s+(-?\d+\.\d+)\s+Eh"),
    "Gsasa": re.compile(r"-> Gsasa\s+(-?\d+\.\d+)\s+Eh"),
    "Ghb": re.compile(r"-> Ghb\s+(-?\d+\.\d+)\s+Eh"),
    "Gshift": re.compile(r"-> Gshift\s+(-?\d+\.\d+)\s+Eh"),
}
ALPB_KEYS = ("Gsolv", "Gelec", "Gsasa", "Ghb", "Gshift")

# Terms of the xtb CPCM-X free-energy block; the trailing float is kcal/mol.
# xtb 6.7.1 prints "solvation free energy (dG_solv):" with Eh, then kcal/mol.
_CPCMX_RES = {
    "dG_is": re.compile(
        r"Ideal State \(dG_is\):\s+\S+\s+(-?\d+\.\d+)"),
    "dG_av": re.compile(
        r"Averaging correction \(dG_av\):\s+\S+\s+(-?\d+\.\d+)"),
    "dG_res": re.compile(
        r"restoring free energy \(dG_res\):\s+\S+\s+(-?\d+\.\d+)"),
    "dG_CDS": re.compile(
        r"SMD Contribution \(dG_CDS\):\s+\S+\s+(-?\d+\.\d+)"),
    "dG_corr": re.compile(
        r"Standard state correction \(dG_corr\):\s+\S+\s+(-?\d+\.\d+)"),
    "dG_shift": re.compile(
        r"Systematic empirical shift \(dG_shift\)?:?\s+\S+\s+(-?\d+\.\d+)"),
    "Gsolv_cpcmx": re.compile(
        r"solvation free energy(?: \(dG_solv\))?:\s+\S+\s+(-?\d+\.\d+)"),
}
_CPCMX_DG_EH = re.compile(r"total free energy \(dG\)\s+(\S+)")
_CPCMX_GAS_EH = re.compile(r"gas phase energy \(E\)\s+(\S+)")
_FIRST_SUMMARY_E = re.compile(
    r":: total energy\s+(-?\d+\.\d+)\s+Eh")
_GAP_RE = re.compile(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)\s+eV")
CPCMX_KEYS = ("dG_is", "dG_av", "dG_res", "dG_CDS", "dG_corr", "dG_shift",
              "Gsolv_cpcmx")
CPCMX_SIDECAR_KEYS = (
    "dG_res", "dG_CDS", "Gsolv_cpcmx",
    "energy_cpcmx_dg", "energy_firstscf", "energy_gas", "energy_last_total",
    "gap_ab_eV",
)


def parse_alpb_parts(text):
    """GBSA/ALPB solvation terms from the xtb summary, kcal/mol. Missing terms are omitted."""
    parts = {}
    for key, pat in _ALPB_RES.items():
        ms = list(pat.finditer(text))
        if ms:
            parts[key] = float(ms[-1].group(1)) * HARTREE_KCAL
    return parts


def parse_cpcmx_parts(text):
    """CPCM-X free-energy terms in kcal/mol, plus dG, gas and first-SCF energies (Eh)
    and the HOMO-LUMO gap (eV). Missing terms are omitted."""
    parts = {}
    for key, pat in _CPCMX_RES.items():
        ms = list(pat.finditer(text))
        if ms:
            parts[key] = float(ms[-1].group(1))
    m = _CPCMX_DG_EH.search(text)
    if m:
        parts["cpcmx_dG_Eh"] = float(m.group(1))
    m = _CPCMX_GAS_EH.search(text)
    if m:
        parts["cpcmx_Egas_Eh"] = float(m.group(1))
    ms = list(_FIRST_SUMMARY_E.finditer(text))
    if ms:
        parts["E_firstscf_Eh"] = float(ms[0].group(1))
    ms = list(_GAP_RE.finditer(text))
    if ms:
        parts["gap_eV"] = float(ms[-1].group(1))
    return parts


def xtb_env(threads=1):
    n = str(threads)
    env = dict(os.environ)
    env.update({
        "XTBPATH": str(XTB_HOME / "share" / "xtb"),
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "OMP_STACKSIZE": "4G",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    })
    return env


def _element(line):
    e = line[76:78].strip()
    if e:
        return e.capitalize()
    name = line[12:16].strip()
    letters = "".join(c for c in name if c.isalpha())
    return (letters[:1] or "X").capitalize()


def split_chains(capped_pdb):
    """(chain_A_atoms, chain_B_atoms) as [(element, x, y, z), ...]."""
    a, b = [], []
    for line in Path(capped_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rec = (_element(line), float(line[30:38]), float(line[38:46]), float(line[46:54]))
        (a if line[21] == "A" else b).append(rec)
    return a, b


def write_xyz(atoms, path, comment=""):
    lines = [str(len(atoms)), comment]
    lines += [f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, x, y, z in atoms]
    Path(path).write_text("\n".join(lines) + "\n")


def _unlimited_stack():
    """Remove the stack limit for the xtb child process.

    xtb can segfault in its OpenMP coordination-number loop with a bounded stack
    (xtb_disp_coordina/coordinationnumber.f90)."""
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


def run_sp(xyz, charge, workdir, env, solvent="water", gfn=2, accuracy=1.0,
           solvation="alpb"):
    """Single-point energy. Returns (energy in Eh, err); energy is None on failure."""
    rec, err = run_sp_detailed(xyz, charge, workdir, env, solvent, gfn, accuracy,
                               solvation)
    if rec is None:
        return None, err
    return rec["E_Eh"], ""


#: xtb command-line flag for each implicit-solvent model. ddCOSMO is polar-only:
#: unlike GBSA and ALPB it has no surface-area or hydrogen-bond term.
SOLVATION_FLAGS = {"alpb": "--alpb", "gbsa": "--gbsa", "cosmo": "--cosmo",
                   "cpcmx": "--cpcmx"}


def solvation_flag(solvation):
    try:
        return SOLVATION_FLAGS[solvation]
    except KeyError:
        raise ValueError(
            f"unknown solvation model {solvation!r}; "
            f"expected one of {sorted(SOLVATION_FLAGS)}") from None


def run_sp_detailed(xyz, charge, workdir, env, solvent="water", gfn=2, accuracy=1.0,
                    solvation="alpb", timeout=None):
    """Single point with parsed output. Returns (record, err): energy in Eh plus
    implicit-solvent terms in kcal/mol, or (None, err) on failure."""
    if timeout is None:
        timeout = 14400 if solvation == "cpcmx" else 3600
    cmd = [str(XTB_EXE), Path(xyz).name, "--sp", "--gfn", str(gfn),
           "--chrg", str(int(charge)), "--acc", str(accuracy), "--norestart"]
    if solvent:
        cmd += [solvation_flag(solvation), solvent]
    try:
        p = subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=_unlimited_stack)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    out = p.stdout + p.stderr
    stem = Path(xyz).stem
    try:
        (Path(workdir) / f"{stem}.xtbout").write_text(out)
    except OSError:
        pass
    m = None
    for m in _ENERGY_RE.finditer(out):
        pass
    if m is None:
        tail = "\n".join(out.strip().splitlines()[-4:])
        return None, tail[:200]
    rec = {"E_Eh": float(m.group(1)), **parse_alpb_parts(out),
           **parse_cpcmx_parts(out)}
    return rec, ""


def write_xcontrol_fix(path, freeze_serials):
    """Write an xcontrol $fix block that freezes the given atoms during optimisation."""
    from interaction import to_atom_ranges
    ranges = to_atom_ranges(freeze_serials)
    Path(path).write_text(f"$fix\n  atoms: {ranges}\n$end\n")


def run_opt(xyz, charge, workdir, env, solvent="water", gfn=2, accuracy=1.0,
            solvation="alpb", freeze=None, opt_level="normal", cycles=100,
            timeout=7200):
    """Optimize xyz in workdir. Returns (xtbopt.xyz Path, err)."""
    workdir = Path(workdir)
    xyz = Path(xyz)
    cmd = [str(XTB_EXE), xyz.name, "--opt", opt_level, "--gfn", str(gfn),
           "--chrg", str(int(charge)), "--acc", str(accuracy),
           "--norestart", "--cycles", str(cycles)]
    if solvent:
        cmd += [solvation_flag(solvation), solvent]
    if freeze:
        write_xcontrol_fix(workdir / "xtb.inp", freeze)
        cmd += ["--input", "xtb.inp"]
    try:
        p = subprocess.run(cmd, cwd=str(workdir), env=env, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=_unlimited_stack)
    except subprocess.TimeoutExpired:
        return None, "opt_timeout"
    out = p.stdout + p.stderr
    (workdir / "xtb_opt.log").write_text(out)
    opt_xyz = workdir / "xtbopt.xyz"
    if opt_xyz.exists() and opt_xyz.stat().st_size > 0:
        return opt_xyz, ""
    tail = "\n".join(out.strip().splitlines()[-8:])
    return None, tail[:300]


def interaction_energy(capped_pdb, workdir, charge_a, charge_b, solvent="water",
                       gfn=2, threads=1, accuracy=1.0, solvation="alpb"):
    """GFN-xTB interaction energy in kcal/mol: E(AB) - E(A) - E(B).

    Returns (energy, status). Charges are the formal per-partner charges from
    truncate_interface.py. A species whose SCF fails is retried at charge offsets
    -2, +2, -4, +4 as in the MOPAC protocol; even offsets keep the electron count even.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = xtb_env(threads)
    atoms_a, atoms_b = split_chains(capped_pdb)
    if not atoms_a or not atoms_b:
        return None, "missing_chain"

    write_xyz(atoms_a + atoms_b, workdir / "ab.xyz", "complex")
    write_xyz(atoms_a, workdir / "a.xyz", "partner A + waters")
    write_xyz(atoms_b, workdir / "b.xyz", "partner B")

    def _try(name, xyz, q):
        last_err = ""
        for dq in (0, -2, 2, -4, 4):
            rec, err = run_sp_detailed(xyz, q + dq, workdir, env, solvent, gfn,
                                       accuracy, solvation)
            if rec is not None:
                return rec, q + dq, ""
            last_err = err
            if err == "timeout":
                break
        return None, q, f"{name}:{last_err}"

    rec_a, qa, err = _try("A", "a.xyz", charge_a)
    if rec_a is None:
        return None, f"scf_fail_{err}"
    rec_b, qb, err = _try("B", "b.xyz", charge_b)
    if rec_b is None:
        return None, f"scf_fail_{err}"
    rec_ab, _, err = _try("AB", "ab.xyz", qa + qb)
    if rec_ab is None:
        return None, f"scf_fail_{err}"

    kcal = (rec_ab["E_Eh"] - rec_a["E_Eh"] - rec_b["E_Eh"]) * HARTREE_KCAL
    status = "ok" if (qa, qb) == (charge_a, charge_b) else "charge_retry_ok"
    return kcal, status


def interaction_decomp(capped_pdb, workdir, charge_a, charge_b, solvent="water",
                       gfn=2, threads=1, accuracy=1.0, solvation="alpb"):
    """Like interaction_energy, plus interaction differences of the solvation terms.

    Returns (record, status). Besides energy, record holds dGsolv, dGelec, dGsasa,
    dGhb and dGshift (GBSA/ALPB) or the CPCM-X terms, in kcal/mol, when all three
    single points printed them.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = xtb_env(threads)
    atoms_a, atoms_b = split_chains(capped_pdb)
    if not atoms_a or not atoms_b:
        return None, "missing_chain"

    write_xyz(atoms_a + atoms_b, workdir / "ab.xyz", "complex")
    write_xyz(atoms_a, workdir / "a.xyz", "partner A + waters")
    write_xyz(atoms_b, workdir / "b.xyz", "partner B")

    def _try(name, xyz, q):
        last_err = ""
        for dq in (0, -2, 2, -4, 4):
            rec, err = run_sp_detailed(xyz, q + dq, workdir, env, solvent, gfn,
                                       accuracy, solvation)
            if rec is not None:
                return rec, q + dq, ""
            last_err = err
            if err == "timeout":
                break
        return None, q, f"{name}:{last_err}"

    rec_a, qa, err = _try("A", "a.xyz", charge_a)
    if rec_a is None:
        return None, f"scf_fail_{err}"
    rec_b, qb, err = _try("B", "b.xyz", charge_b)
    if rec_b is None:
        return None, f"scf_fail_{err}"
    rec_ab, _, err = _try("AB", "ab.xyz", qa + qb)
    if rec_ab is None:
        return None, f"scf_fail_{err}"

    energy_last = (rec_ab["E_Eh"] - rec_a["E_Eh"] - rec_b["E_Eh"]) * HARTREE_KCAL
    out = {"energy": energy_last, "energy_last_total": energy_last}

    def _delta_eh(key):
        if key in rec_ab and key in rec_a and key in rec_b:
            return (rec_ab[key] - rec_a[key] - rec_b[key]) * HARTREE_KCAL
        return None

    e_1st = _delta_eh("E_firstscf_Eh")
    if e_1st is not None:
        out["energy_firstscf"] = e_1st
    if "gap_eV" in rec_ab:
        out["gap_ab_eV"] = rec_ab["gap_eV"]
    # xtb 6.7.1 prints "total free energy (dG)" with few digits; the last TOTAL ENERGY
    # equals it and is used as the full-precision CPCM-X dG.
    out["energy_cpcmx_dg"] = energy_last

    for key in ALPB_KEYS:
        if key in rec_ab and key in rec_a and key in rec_b:
            out[f"d{key}"] = rec_ab[key] - rec_a[key] - rec_b[key]
    if "dGsasa" in out and "dGhb" in out:
        out["dCDS"] = out["dGsasa"] + out["dGhb"]
    for key in CPCMX_KEYS:
        if key in rec_ab and key in rec_a and key in rec_b:
            out[key] = rec_ab[key] - rec_a[key] - rec_b[key]
    if "Gsolv_cpcmx" in out:
        out["energy_gas"] = energy_last - out["Gsolv_cpcmx"]
    status = "ok" if (qa, qb) == (charge_a, charge_b) else "charge_retry_ok"
    return out, status


def cleanup(workdir):
    shutil.rmtree(workdir, ignore_errors=True)
