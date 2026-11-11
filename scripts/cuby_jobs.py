#!/usr/bin/env python3
"""cuby4 YAML construction and MOZYME charge handling for capped interface shells.

Writes the interaction single-point job for a shell from truncate_interface.py
and an SQM optimization job in which only interface side chains and interfacial
waters move (optimize_region), while the capped backbone and caps stay fixed.
Method, correction and solvent settings come from interaction.SQM_PRESETS.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from interaction import SQM_PRESETS, PM6_ML_EXE, to_atom_ranges, write_modifier_blocks


def _preset_block(f, preset, cosmo_nspa=None):
    """Write the shared MOPAC method/correction/solvent lines for a preset.

    ``cosmo_nspa`` (if set) adds ``mopac_cosmo_nspa: N`` for presets with an
    implicit-solvent model; more COSMO segments per atom reduce numerical noise
    in the solvation energy.
    """
    p = SQM_PRESETS[preset]
    f.write(f"method: {p['method']}\n")
    if p.get("mopac_corrections"):
        f.write(f"mopac_corrections: {p['mopac_corrections']}\n")
    if p.get("solvent_model"):
        f.write(f"solvent_model: {p['solvent_model']}\n")
        if p.get("solvent_epsilon") is not None:
            f.write(f"solvent_epsilon: {p['solvent_epsilon']}\n")
        if p["solvent_model"] == "cosmo2" and p.get("mopac_cosmo2_surf_tension") is not None:
            f.write(f"mopac_cosmo2_surf_tension: {p['mopac_cosmo2_surf_tension']}\n")
        if cosmo_nspa is not None:
            f.write(f"mopac_cosmo_nspa: {cosmo_nspa}\n")
    if p.get("modifiers"):
        f.write(f"modifiers: {p['modifiers']}\n")
    f.write(f'mopac_keywords: "{p["mopac_keywords"]}"\n')
    write_modifier_blocks(f, p)


def resolve_mopac_exe(preset, default_exe):
    """PM6-ML presets must run through the mopac-ml-cuby launcher."""
    if SQM_PRESETS[preset].get("is_pm6_ml"):
        return PM6_ML_EXE
    return default_exe


def _chain_atom_ranges(capped_pdb):
    """1-based atom serials for chain A and chain B (file order) of a capped shell.

    Waters are written with chain A and therefore belong to molecule_a.
    """
    a, b = [], []
    idx = 0
    for line in Path(capped_pdb).read_text().splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            idx += 1
            (a if line[21] == "A" else b).append(idx)
    return a, b


WATER_RESNAMES_YAML = {"HOH", "WAT", "TP3", "TIP3", "SOL", "OPC", "T3P"}


def _chain_atom_ranges_partitioned(capped_pdb):
    """Chain split with each water assigned to the nearer protein chain.

    With all waters in molecule_a, only water-receptor terms cancel in
    E(A+W+B) - E(A+W) - E(B), and water-ligand terms remain in the interaction
    energy. Assigning each water to its nearer partner cancels the dominant
    water term for every water. Water is neutral, so fragment charges and
    electron parity are unchanged.
    """
    prot_a, prot_b, waters = [], [], {}
    idx = 0
    for line in Path(capped_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        idx += 1
        resname = line[17:20].strip().upper()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if resname in WATER_RESNAMES_YAML:
            waters.setdefault(line[21:27], []).append((idx, xyz))
        elif line[21] == "A":
            prot_a.append(xyz)
        else:
            prot_b.append(xyz)
    if not prot_a or not prot_b:
        raise ValueError(f"{capped_pdb}: need both chain A and chain B protein atoms")

    def min_d2(xyz, pts):
        return min((xyz[0] - p[0]) ** 2 + (xyz[1] - p[1]) ** 2
                   + (xyz[2] - p[2]) ** 2 for p in pts)

    a_atoms, b_atoms = [], []
    idx = 0
    for line in Path(capped_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        idx += 1
        if line[17:20].strip().upper() in WATER_RESNAMES_YAML:
            continue
        (a_atoms if line[21] == "A" else b_atoms).append(idx)
    for key, atoms in waters.items():
        da = min(min_d2(xyz, prot_a) for _, xyz in atoms)
        db = min(min_d2(xyz, prot_b) for _, xyz in atoms)
        target = a_atoms if da <= db else b_atoms
        target.extend(i for i, _ in atoms)
    return sorted(a_atoms), sorted(b_atoms)


_COMPUTED_Q_RE = re.compile(r"COMPUTED CHARGE ON SYSTEM:\s*([+-]?\d+)")

_FRAG_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12,
           "P": 15, "S": 16, "CL": 17, "K": 19, "CA": 20, "FE": 26, "ZN": 30,
           "SE": 34, "BR": 35, "I": 53}


def fragment_nuclear_charges(capped_pdb, water_assign="receptor"):
    """(sum_z_a, sum_z_b) for a capped shell, used for per-fragment parity checks.

    Computed from the geometry when the sidecar does not store these values.
    Returns (None, None) if an element is not recognised.
    """
    a_idx, b_idx = (_chain_atom_ranges_partitioned(capped_pdb)
                    if water_assign == "partition"
                    else _chain_atom_ranges(capped_pdb))
    a_set, b_set = set(a_idx), set(b_idx)
    za = zb = 0
    idx = 0
    for line in Path(capped_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        idx += 1
        el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        z = _FRAG_Z.get(el)
        if z is None:
            return None, None
        if idx in a_set:
            za += z
        elif idx in b_set:
            zb += z
    return za, zb


def mozyme_computed_charges(workdir):
    """Charges computed by MOZYME, read from a cuby workdir after a failed attempt.

    When the submitted charge disagrees with MOZYME's Lewis structure, MOPAC
    stops during Lewis setup, before any SCF, and prints the charge it computed.
    cuby aborts the interaction job at the first failing fragment, so one attempt
    usually reveals only ``ab``; callers accumulate results across attempts. The
    submitted charge must be plausible: for an implausible charge MOPAC prints
    only a generic error without a computed charge.

    Returns a dict with any of the keys "a", "b", "ab".
    """
    found = {}
    for job, key in (("job_a_MOPAC", "a"), ("job_b_MOPAC", "b"),
                     ("job_ab_MOPAC", "ab"),
                     # cuby optimize uses a single complex job
                     ("job_MOPAC", "ab")):
        out = Path(workdir) / job / "mopac.in.out"
        if not out.exists():
            continue
        m = _COMPUTED_Q_RE.search(out.read_text(errors="ignore"))
        if m:
            found[key] = int(m.group(1))
    return found


def next_charge_guess(formal_a, formal_b, sum_z, known, tried,
                      sum_z_a=None, sum_z_b=None, extra_seeds=()):
    """Next (qa, qb) to submit, given the charges MOZYME has reported so far.

    ``known`` is the accumulated dict from mozyme_computed_charges. A known
    complex charge restricts candidates to pairs with that sum; a known fragment
    charge fixes that fragment. Otherwise shifts of up to 4 around the seed
    charges are tried.

    ``sum_z_a``/``sum_z_b`` enable per-fragment parity filtering: cuby runs job_a
    and job_b as independent closed-shell singlets, so a pair with an even total
    can still be rejected for one fragment ("Charge specified on input not
    possible for closed-shell system").

    Returns None when all candidates are exhausted.
    """
    qa_known, qb_known, qab_known = known.get("a"), known.get("b"), known.get("ab")

    def _ok(pair):
        qa, qb = pair
        if pair in tried:
            return False
        if sum_z_a is not None and (int(sum_z_a) - qa) % 2 != 0:
            return False
        if sum_z_b is not None and (int(sum_z_b) - qb) % 2 != 0:
            return False
        if sum_z is not None and (int(sum_z) - (qa + qb)) % 2 != 0:
            return False
        return True

    exact = []
    if qa_known is not None and qb_known is not None:
        exact.append((qa_known, qb_known))
    if qa_known is not None and qab_known is not None:
        exact.append((qa_known, qab_known - qa_known))
    if qb_known is not None and qab_known is not None:
        exact.append((qab_known - qb_known, qb_known))
    for pair in exact:
        if _ok(pair):
            return pair

    def _consistent(pair):
        if qab_known is not None and sum(pair) != qab_known:
            return False
        if qa_known is not None and pair[0] != qa_known:
            return False
        if qb_known is not None and pair[1] != qb_known:
            return False
        return True

    # Shifts include odd values: when a fragment's formal charge sits on the
    # wrong parity, every even-only shift is rejected on that fragment.
    shifts = (0, -1, 1, -2, 2, -3, 3, -4, 4)
    for seed_a, seed_b in tuple(extra_seeds) + ((formal_a, formal_b),):
        for dq_b in shifts:
            for dq_a in shifts:
                pair = (seed_a + dq_a, seed_b + dq_b)
                if _consistent(pair) and _ok(pair):
                    return pair
    return None


def mopac_charge_candidates(charge_a, charge_b, sum_z=None, extra_seeds=()):
    """Yield (qa, qb): formal charges first, then small even-electron shifts.

    MOZYME may reject the formal charges of a capped shell; a ±2 shift on one
    partner usually matches MOPAC's Lewis-structure charge.

    ``extra_seeds`` prepends alternate (qa, qb) pairs (e.g. truncation formal
    charges when the tleap netCharge differs) before the primary seed shifts.
    """
    seen = set()

    def _yield_seed(seed_a, seed_b):
        for dq_b in (0, -2, 2, -4, 4):
            for dq_a in (0, -2, 2):
                qa, qb = seed_a + dq_a, seed_b + dq_b
                if (qa, qb) in seen:
                    continue
                seen.add((qa, qb))
                if sum_z is not None and (int(sum_z) - (qa + qb)) % 2 != 0:
                    continue
                yield qa, qb

    for sa, sb in extra_seeds:
        yield from _yield_seed(sa, sb)
    yield from _yield_seed(charge_a, charge_b)


def write_interaction_yaml(out_yaml, capped_pdb, preset, mopac_exe, parallel,
                           charge_a, charge_b, cosmo_nspa=None,
                           water_assign="receptor"):
    """Write a cuby4 interaction YAML for the capped shell with explicit charges.

    ``charge_a``/``charge_b`` include chain-terminus charges from
    truncate_interface.py; interaction.get_pdb_chains counts side-chain charges
    only, which is incorrect for a truncated interface.
    """
    if water_assign == "partition":
        a_atoms, b_atoms = _chain_atom_ranges_partitioned(capped_pdb)
    else:
        a_atoms, b_atoms = _chain_atom_ranges(capped_pdb)
    if not a_atoms or not b_atoms:
        raise ValueError(f"{capped_pdb}: need both chain A and chain B atoms")
    mopac_exe = resolve_mopac_exe(preset, mopac_exe)
    with open(out_yaml, "w") as f:
        f.write("job: interaction\n")
        f.write(f"geometry: {Path(capped_pdb).name}\n")
        f.write("geometry_setup_from_file: yes\n\n")
        f.write("interface: mopac\n")
        _preset_block(f, preset, cosmo_nspa=cosmo_nspa)
        f.write(f"mopac_exe: {mopac_exe}\n")
        f.write(f"parallel: {parallel}\n")
        f.write("job_cleanup: no\n")
        f.write("\nmolecule_a:\n")
        f.write(f'  selection: "@{to_atom_ranges(a_atoms)}"\n')
        f.write(f"  charge: {charge_a}\n")
        f.write("\nmolecule_b:\n")
        f.write(f'  selection: "@{to_atom_ranges(b_atoms)}"\n')
        f.write(f"  charge: {charge_b}\n")
        f.write("\nmolecule_ab:\n")
        f.write(f"  charge: {charge_a + charge_b}\n")


def write_optimize_yaml(out_yaml, geometry_name, preset, mopac_exe, parallel,
                        mobile_indices, total_charge, restart_pdb="optimized.pdb",
                        optimizer="lbfgs", opt_quality=0.5, maxcycles=100, cosmo_nspa=None):
    """Write a cuby4 ``optimize`` YAML that relaxes only ``mobile_indices``.

    ``mobile_indices`` (1-based atom serials) define ``optimize_region``; all
    other atoms (capped backbone, caps) are fixed. ``total_charge`` is the net
    charge of the capped complex (charge_a + charge_b).
    """
    mopac_exe = resolve_mopac_exe(preset, mopac_exe)
    region = to_atom_ranges(mobile_indices)
    with open(out_yaml, "w") as f:
        f.write("job: optimize\n")
        f.write(f"geometry: {geometry_name}\n\n")
        # No 'geometry_setup_from_file': cuby would read the (absent) PDB charge
        # as 0 and MOPAC would fail its "SINGLET with odd number of electrons"
        # check. The total charge is set explicitly below.
        f.write("interface: mopac\n")
        _preset_block(f, preset, cosmo_nspa=cosmo_nspa)
        f.write(f"mopac_exe: {mopac_exe}\n")
        f.write(f"charge: {total_charge}\n")
        f.write("mopac_precise: yes\n")
        f.write(f"parallel: {parallel}\n")
        f.write("job_cleanup: no\n\n")
        f.write(f"optimizer: {optimizer}\n")
        f.write(f"opt_quality: {opt_quality}\n")
        f.write("coordinates: cartesian\n")
        f.write(f"maxcycles: {maxcycles}\n")
        if region:
            f.write(f'optimize_region: "@{region}"\n')
        f.write(f"restart_file: {restart_pdb}\n")
        f.write("history_freq: 0\n")


def overlay_xyz_onto_pdb(template_pdb, coords_file, out_pdb):
    """Write ``template_pdb`` to ``out_pdb`` with coordinates (in atom order) from
    ``coords_file`` (.xyz or .pdb from a cuby4 optimization).

    Chain IDs, residue names and element columns are taken from the template,
    since the optimizer output does not reliably preserve them.
    """
    coords = _read_coords(coords_file)
    out_lines = []
    ai = 0
    for line in Path(template_pdb).read_text().splitlines(keepends=True):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if ai >= len(coords):
                raise ValueError(
                    f"coord count {len(coords)} < atoms in {template_pdb}")
            x, y, z = coords[ai]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            ai += 1
        out_lines.append(line)
    if ai != len(coords):
        raise ValueError(
            f"atom/coord mismatch: pdb has {ai}, coords file has {len(coords)}")
    Path(out_pdb).write_text("".join(out_lines))


def _read_coords(coords_file):
    coords_file = Path(coords_file)
    txt = coords_file.read_text().splitlines()
    if coords_file.suffix == ".pdb":
        out = []
        for line in txt:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        return out
    # XYZ: first line = count, second = comment, then "El x y z"
    out = []
    try:
        n = int(txt[0].strip())
        body = txt[2:2 + n]
    except (ValueError, IndexError):
        body = [l for l in txt if len(l.split()) >= 4]
    for line in body:
        parts = line.split()
        if len(parts) >= 4:
            out.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return out
