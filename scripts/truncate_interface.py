#!/usr/bin/env python3
"""Two-sided truncation of a protein-protein interface with backbone capping.

Builds a capped interface-shell PDB from one MD snapshot: residues of each partner
within ``cutoff`` A (heavy atoms) of the other partner, or from a dASA mask, plus
PRO/GLY/disulfide pull-in; interior backbone cuts are capped and true termini kept.
Writes chain A (partner A + interfacial waters), chain B, and a .trunc.json sidecar.

Usage: truncate_interface.py --pdb wet_f1.pdb --out shell.pdb --input-pdb complex.pdb
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Formal-charge table shared with interaction.py.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from interaction import RESIDUE_CHARGES  # noqa: E402

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID",
    "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "CYX", "CYM", "CYP", "ASH", "GLH", "LYN",
}
WATER_RESNAMES = {"HOH", "WAT", "TP3", "TIP3", "TIP4", "SOL", "OPC", "T3P"}
ION_ELEMENTS = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "CO", "NI", "BR", "IOD"}
CAP_RESNAMES = {"BCB", "BCC", "BCG", "BNC", "ACE", "NME"}

# Polar residues that anchor interfacial waters (POLAR setup of Nwat-MMGBSA,
# Maffucci & Contini, 2016; see nwat_mmgbsa.py).
POLAR_RESNAMES = {
    "ARG", "ASH", "ASP", "GLH", "GLN", "GLU", "HID", "HIE", "HIP",
    "LYN", "LYS", "SER", "THR", "TRP", "TYR",
}

# Backbone and terminal-H atom names kept frozen in SQM optimizations; all other
# atoms of standard residues are treated as mobile side-chain atoms.
BACKBONE_ATOMS = {
    "N", "H", "H1", "H2", "H3", "CA", "HA", "HA2", "HA3", "C", "O", "OXT", "OC1", "OC2",
}

# Tolerance (A) added to distance cutoffs to absorb floating-point noise at the
# boundary.
CUTOFF_EPS = 0.01

# Atomic numbers for the electron-parity check. A closed-shell singlet needs an
# even electron count (sum_Z - charge); MOPAC rejects odd counts.
_ATOMIC_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "F": 9, "CL": 17, "BR": 35,
    "I": 53, "NA": 11, "K": 19, "MG": 12, "CA": 20, "ZN": 30, "FE": 26, "MN": 25,
    "CU": 29, "CO": 27, "NI": 28, "SE": 34,
}

# Cap construction recipes
_H_BOND_FRACTION = 0.72


def _h_toward(parent, heavy):
    return tuple(p + (h - p) * _H_BOND_FRACTION for p, h in zip(parent, heavy))


def _away(origin, other, dist):
    """Point ``dist`` A from origin, opposite ``other`` (cap methyl placement)."""
    d = [oi - ot for oi, ot in zip(origin, other)]
    norm = math.sqrt(sum(c * c for c in d))
    if norm < 0.2:
        return origin
    return tuple(oi + (c / norm) * dist for oi, c in zip(origin, d))


def _place_amide_h(src):
    """Position of the backbone amide H on N.

    Uses the existing H if present; otherwise (e.g. proline, which has no amide
    H) places one 1.01 A from N, pointing away from CA, so the C-terminal cap
    remains a closed-shell N-substituted amide."""
    h = src.get("H")
    if h is not None:
        return h
    n, ca = src.get("N"), src.get("CA")
    if n is None or ca is None:
        return None
    d = [ni - ci for ni, ci in zip(n, ca)]
    norm = math.sqrt(sum(c * c for c in d)) or 1.0
    return tuple(ni + (c / norm) * 1.01 for ni, c in zip(n, d))


_CAP_RECIPES = {
    "BNC": [
        ("HA", ("h_toward", "C", "CA")),
        ("C", ("copy", "C")),
        ("O", ("copy", "O")),
    ],
    "BCC": [
        ("N", ("copy", "N")),
        ("H", ("amide_h",)),
        ("CA", ("copy", "CA")),
        ("HA", ("copy", "HA")),
        ("HB", ("h_toward", "CA", "CB")),
        ("H2", ("h_toward", "CA", "C")),
    ],
    "BCG": [
        ("N", ("copy", "N")),
        ("H", ("amide_h",)),
        ("CA", ("copy", "CA")),
        ("HA2", ("copy", "HA2")),
        ("HA3", ("copy", "HA3")),
        ("H2", ("h_toward", "CA", "C")),
    ],
    "BCB": [
        ("N", ("copy", "N")),
        ("H", ("amide_h",)),
        ("CA", ("copy", "CA")),
        ("HA", ("copy", "HA")),
        ("HB", ("h_toward", "CA", "CB")),
        ("C", ("copy", "C")),
        ("O", ("copy", "O")),
    ],
    # Standard AMBER ACE/NME caps (ff14SB). They are closed-shell, so each capped
    # partner keeps even electron parity, unlike BNC/BCC caps. Only heavy atoms
    # and the amide H are written; tleap adds the methyl hydrogens.
    "ACE": [                       # N-terminal cap, from the deleted preceding residue
        ("C", ("copy", "C")),
        ("O", ("copy", "O")),
        # Methyl placed opposite the kept residue's N rather than on the deleted
        # CA: when two fragments share one deleted spacer residue, that CA would
        # coincide with the preceding fragment's NME carbon.
        ("CH3", ("away", "C", "KEEP_N", 1.52)),
    ],
    "NME": [                       # C-terminal cap, from the deleted following residue
        ("N", ("copy", "N")),
        ("H", ("amide_h",)),
        # ff14SB NME names the methyl carbon C, not CH3.
        ("C", ("away", "N", "KEEP_C", 1.47)),
    ],
}


def _build_cap_atoms(cap_name, src_atoms):
    out = []
    for atom_name, recipe in _CAP_RECIPES[cap_name]:
        if recipe[0] == "copy":
            xyz = src_atoms.get(recipe[1])
        elif recipe[0] == "amide_h":
            xyz = _place_amide_h(src_atoms)
        elif recipe[0] == "away":
            origin, other = src_atoms.get(recipe[1]), src_atoms.get(recipe[2])
            xyz = (_away(origin, other, recipe[3]) if origin and other
                   else src_atoms.get("CA") or origin)
        else:  # h_toward
            parent, heavy = src_atoms.get(recipe[1]), src_atoms.get(recipe[2])
            xyz = _h_toward(parent, heavy) if parent and heavy else parent
        if xyz is None:
            continue
        out.append((atom_name, xyz, atom_name[0]))
    return out


# PDB input
def _element_of(line, atom_name):
    el = line[76:78].strip() if len(line) >= 78 else ""
    if el:
        return el
    # Fall back to the first character of the atom name.
    nm = atom_name.strip()
    return nm[0] if nm else "X"


def parse_snapshot(pdb_path, n_chain_a, protein_res_count):
    """Parse a dry/wet snapshot into ordered residues tagged by partner.

    Returns ``partners`` = {"A": [resrec...], "B": [...], "W": [...]} where each
    resrec is a dict with resid, resname, partner and an ``atoms`` list of
    ``(name, (x,y,z), element)`` tuples (in file order).
    """
    partners = {"A": [], "B": [], "W": []}
    cur = None
    cur_key = None
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            atom = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element = _element_of(line, atom)

            if resname in WATER_RESNAMES or resseq > protein_res_count:
                partner = "W"
            elif resseq <= n_chain_a:
                partner = "A"
            else:
                partner = "B"

            key = (partner, resseq, line[26:27])  # includes insertion code
            if key != cur_key:
                cur = {
                    "resid": resseq, "resname": resname, "partner": partner,
                    "atoms": [],
                }
                partners[partner].append(cur)
                cur_key = key
            cur["atoms"].append((atom, (x, y, z), element))
    return partners


def _is_true_water(resrec):
    """True for a water residue: contains O and only H/O atoms (excludes ions named as solvent)."""
    elems = {el.upper() for (_a, _xyz, el) in resrec["atoms"]}
    if elems & ION_ELEMENTS:
        return False
    return "O" in elems and elems <= {"O", "H"}


def _heavy_coords(resrec):
    return np.array(
        [xyz for (_n, xyz, el) in resrec["atoms"] if el.upper() != "H"],
        dtype=float,
    )


def _atoms_dict(resrec):
    return {name: xyz for (name, xyz, _el) in resrec["atoms"]}


def _atom_xyz(resrec, name):
    for n, xyz, _el in resrec["atoms"]:
        if n == name:
            return xyz
    return None


def _backbone_continuous(res_a, res_b):
    c = _atom_xyz(res_a, "C")
    n = _atom_xyz(res_b, "N")
    if c is None or n is None:
        return False
    return math.dist(c, n) < 2.0


# Interface selection
def select_interface_dasa(partners, dasa_resids):
    """Select interface residues by global residue number (PyMOL dASA mask)."""
    sel = set()
    for key in ("A", "B"):
        for r in partners[key]:
            if r["resname"] not in STANDARD_AA:
                continue
            if int(r["resid"]) in dasa_resids:
                sel.add(id(r))
    return sel


def _dasa_with_bilateral_fallback(partners, dasa_resids, cutoff):
    """dASA selection, supplemented with distance-based picks on any empty face.

    A dASA mask can be one-sided (e.g. 1ZHI) or mis-split when the A/B boundary
    differs; adding the distance selection for the missing partner keeps both
    partners in the capped shell."""
    sel = select_interface_dasa(partners, dasa_resids)
    aa_a = [r for r in partners["A"] if r["resname"] in STANDARD_AA]
    aa_b = [r for r in partners["B"] if r["resname"] in STANDARD_AA]
    has_a = any(id(r) in sel for r in aa_a)
    has_b = any(id(r) in sel for r in aa_b)
    if has_a and has_b:
        return sel
    dist = select_interface(partners, cutoff)
    if not has_a:
        sel |= {id(r) for r in aa_a if id(r) in dist}
    if not has_b:
        sel |= {id(r) for r in aa_b if id(r) in dist}
    return sel


def select_interface(partners, cutoff):
    """Two-sided heavy-atom selection: a residue of A is selected if any heavy
    atom is within ``cutoff`` of any heavy atom of B, and vice-versa.
    Returns a set of ``id(resrec)`` for selected residues (A and B only).
    """
    aa_a = [r for r in partners["A"] if r["resname"] in STANDARD_AA]
    aa_b = [r for r in partners["B"] if r["resname"] in STANDARD_AA]
    if not aa_a or not aa_b:
        return set()

    heavy_b = [_heavy_coords(r) for r in aa_b]
    heavy_b_all = np.vstack([h for h in heavy_b if len(h)]) if heavy_b else np.zeros((0, 3))
    heavy_a = [_heavy_coords(r) for r in aa_a]
    heavy_a_all = np.vstack([h for h in heavy_a if len(h)]) if heavy_a else np.zeros((0, 3))

    thr = cutoff + CUTOFF_EPS
    sel = set()

    def _min_dist(coords, other_all):
        if len(coords) == 0 or len(other_all) == 0:
            return math.inf
        # pairwise min distance
        d2 = ((coords[:, None, :] - other_all[None, :, :]) ** 2).sum(-1)
        return math.sqrt(float(d2.min()))

    for r, h in zip(aa_a, heavy_a):
        if _min_dist(h, heavy_b_all) <= thr:
            sel.add(id(r))
    for r, h in zip(aa_b, heavy_b):
        if _min_dist(h, heavy_a_all) <= thr:
            sel.add(id(r))
    return sel


def _apply_chemistry_rules(aa_keys, sel):
    """In-place augment ``sel`` (set of id()) for one partner's ordered AA list:
    PRO pull-in, lone-GLY bridge fill, disulfide partners."""
    in_pocket = {id(r): (id(r) in sel) for r in aa_keys}

    # PRO pull-in (iterate to a fixed point).
    changed = True
    while changed:
        changed = False
        for i, r in enumerate(aa_keys):
            if (r["resname"] == "PRO" and not in_pocket[id(r)]
                    and i > 0 and in_pocket[id(aa_keys[i - 1])]):
                in_pocket[id(r)] = True
                changed = True

    # Lone-GLY bridge fill.
    for i, r in enumerate(aa_keys):
        if (r["resname"] == "GLY" and not in_pocket[id(r)]
                and 0 < i < len(aa_keys) - 1
                and in_pocket[id(aa_keys[i - 1])] and in_pocket[id(aa_keys[i + 1])]):
            in_pocket[id(r)] = True

    # Disulfide partners travel together.
    cyx = [r for r in aa_keys if r["resname"] in {"CYX", "CYS", "CYM", "CYP"}]
    for r in cyx:
        if not in_pocket[id(r)]:
            continue
        sg1 = _atom_xyz(r, "SG")
        if sg1 is None:
            continue
        for other in cyx:
            if other is r:
                continue
            sg2 = _atom_xyz(other, "SG")
            if sg2 is not None and math.dist(sg1, sg2) < 2.5:
                in_pocket[id(other)] = True

    for r in aa_keys:
        if in_pocket[id(r)]:
            sel.add(id(r))


# PDB output
def _format_atom_name(atom, element):
    if len(atom) >= 4:
        return atom[:4]
    if len(element.strip()) == 2:
        return f"{atom:<4s}"
    return f" {atom:<3s}"


class _Emitter:
    def __init__(self):
        self.lines = []
        self.atom_num = 0   # PDB serial counter (incremented for ATOM and TER)
        self.n_atoms = 0    # ATOM-only counter (cuby4 selections ignore TER)
        self.resid = 0
        # 1-based indices of mobile atoms for SQM optimization, counted over ATOM
        # records only because cuby4 atom selections ignore TER records.
        self.mobile = []

    def _emit(self, atom, resname, x, y, z, element, chain, mobile):
        self.atom_num += 1
        self.n_atoms += 1
        if mobile:
            self.mobile.append(self.n_atoms)
        name = _format_atom_name(atom, element)
        serial = self.atom_num % 100000
        self.lines.append(
            f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain:1s}{self.resid % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
        )

    def residue(self, resrec, chain, mobile_sidechains):
        self.resid += 1
        resname = resrec["resname"]
        for atom, (x, y, z), element in resrec["atoms"]:
            is_mobile = (
                mobile_sidechains
                and resname in STANDARD_AA
                and atom.strip().upper() not in BACKBONE_ATOMS
            )
            self._emit(atom, resname, x, y, z, element, chain, is_mobile)

    def water(self, resrec, chain, mobile):
        self.resid += 1
        for atom, (x, y, z), element in resrec["atoms"]:
            self._emit(atom, "HOH", x, y, z, element, chain, mobile)

    def cap(self, cap_name, src_atoms, chain):
        # Caps are always frozen (they stand in for deleted backbone).
        self.resid += 1
        for atom, (x, y, z), element in _build_cap_atoms(cap_name, src_atoms):
            self._emit(atom, cap_name, x, y, z, element, chain, False)

    def ter(self):
        self.atom_num += 1
        self.lines.append(f"TER   {self.atom_num % 100000:5d}\n")


def _emit_partner(em, aa_keys, sel, chain, mobile_sidechains, caps="single"):
    """Emit one partner's selected fragments with caps.

    ``caps='ace_nme'`` uses standard AMBER ACE/NME caps and no BCB bridging;
    ``caps='single'`` uses the BNC/BCC/BCG/BCB caps."""
    ace_nme = caps == "ace_nme"
    sel_flags = [id(r) in sel for r in aa_keys]
    n = len(aa_keys)

    # Fragment index ranges: maximal runs of selected + peptide-bonded residues.
    frags = []
    i = 0
    while i < n:
        if sel_flags[i]:
            j = i
            while (j + 1 < n and sel_flags[j + 1]
                   and _backbone_continuous(aa_keys[j], aa_keys[j + 1])):
                j += 1
            frags.append((i, j))
            i = j + 1
        else:
            i += 1

    def _is_bcb_gap(e, next_start):
        if next_start - e - 1 != 1:
            return False
        bridge = aa_keys[e + 1]
        if bridge["resname"] == "GLY":
            return False
        return (_backbone_continuous(aa_keys[e], bridge)
                and _backbone_continuous(bridge, aa_keys[next_start]))

    for fi, (s, e) in enumerate(frags):
        first_key, last_key = aa_keys[s], aa_keys[e]

        # N-terminal cap
        prev_bonded = s > 0 and _backbone_continuous(aa_keys[s - 1], first_key)
        if prev_bonded:
            if ace_nme:
                src = _atoms_dict(aa_keys[s - 1])
                src["KEEP_N"] = _atoms_dict(first_key).get("N")
                em.cap("ACE", src, chain)
            else:
                prev_frag = frags[fi - 1] if fi > 0 else None
                absorbed = prev_frag is not None and _is_bcb_gap(prev_frag[1], s)
                if not absorbed:
                    em.cap("BNC", _atoms_dict(aa_keys[s - 1]), chain)
        # At a true terminus the charged N-terminus is kept without a cap.

        # Fragment residues
        for idx in range(s, e + 1):
            em.residue(aa_keys[idx], chain, mobile_sidechains)

        # C-terminal cap
        ended_bcb = False
        next_frag = frags[fi + 1] if fi + 1 < len(frags) else None
        if ace_nme:
            if e + 1 < n and _backbone_continuous(last_key, aa_keys[e + 1]):
                src = _atoms_dict(aa_keys[e + 1])
                src["KEEP_C"] = _atoms_dict(last_key).get("C")
                em.cap("NME", src, chain)
        elif next_frag is not None and _is_bcb_gap(e, next_frag[0]):
            em.cap("BCB", _atoms_dict(aa_keys[e + 1]), chain)
            ended_bcb = True
        elif e + 1 < n and _backbone_continuous(last_key, aa_keys[e + 1]):
            first_deleted = aa_keys[e + 1]["resname"]
            cap_name = "BCG" if first_deleted == "GLY" else "BCC"
            em.cap(cap_name, _atoms_dict(aa_keys[e + 1]), chain)
        if not ended_bcb:
            em.ter()
    return frags


def _interfacial_waters(partners, sel_a_b, cutoff, max_waters=None, polar_waters=False):
    """Waters whose O is within ``cutoff`` A of selected interface heavy atoms.

    By default any selected interface residue anchors waters. With
    ``polar_waters=True`` only polar interface residues (POLAR_RESNAMES) anchor
    waters, as in the cpptraj ``closest Nwat :polar_interface`` selection of the
    Nwat-MMGBSA protocol. Waters bridging both faces are ranked first."""
    sel_heavy_a, sel_heavy_b = [], []
    for r in partners["A"]:
        if id(r) in sel_a_b:
            if polar_waters and r["resname"] not in POLAR_RESNAMES:
                continue
            h = _heavy_coords(r)
            if len(h):
                sel_heavy_a.append(h)
    for r in partners["B"]:
        if id(r) in sel_a_b:
            if polar_waters and r["resname"] not in POLAR_RESNAMES:
                continue
            h = _heavy_coords(r)
            if len(h):
                sel_heavy_b.append(h)
    A = np.vstack(sel_heavy_a) if sel_heavy_a else np.zeros((0, 3))
    B = np.vstack(sel_heavy_b) if sel_heavy_b else np.zeros((0, 3))
    thr2 = (cutoff + CUTOFF_EPS) ** 2
    scored = []
    for w in partners["W"]:
        if not _is_true_water(w):
            continue
        o = _atom_xyz(w, "O")
        if o is None:
            o = w["atoms"][0][1]
        o = np.array(o)
        da = math.sqrt(((A - o) ** 2).sum(1).min()) if len(A) else math.inf
        db = math.sqrt(((B - o) ** 2).sum(1).min()) if len(B) else math.inf
        if (len(A) and da * da <= thr2) or (len(B) and db * db <= thr2):
            scored.append((max(da, db), w))  # bridging score: distance to the farther face
    scored.sort(key=lambda t: t[0])
    if max_waters is not None:
        scored = scored[:max_waters]
    return [w for _s, w in scored]


def _closest_waters(partners, sel_a_b, max_waters, polar_waters=False):
    """N waters closest to selected interface heavy atoms (min distance, no cutoff).

    Used in ``dasa_union`` mode, where the residue set is fixed in advance.
    """
    heavies = []
    for key in ("A", "B"):
        for r in partners[key]:
            if id(r) not in sel_a_b:
                continue
            if polar_waters and r["resname"] not in POLAR_RESNAMES:
                continue
            h = _heavy_coords(r)
            if len(h):
                heavies.append(h)
    H = np.vstack(heavies) if heavies else np.zeros((0, 3))
    scored = []
    for w in partners["W"]:
        if not _is_true_water(w):
            continue
        o = _atom_xyz(w, "O")
        if o is None:
            o = w["atoms"][0][1]
        o = np.array(o)
        d = math.sqrt(((H - o) ** 2).sum(1).min()) if len(H) else math.inf
        scored.append((d, w))
    scored.sort(key=lambda t: t[0])
    if max_waters is not None:
        scored = scored[:max_waters]
    return [w for _s, w in scored]


def residues_near_waters(partners, waters, radius=4.0):
    """Residue numbers with any heavy atom within ``radius`` A of a kept water O."""
    os_ = []
    for w in waters:
        o = _atom_xyz(w, "O")
        if o is None:
            o = w["atoms"][0][1]
        os_.append(np.array(o))
    if not os_:
        return set()
    O = np.vstack(os_)
    thr = radius + CUTOFF_EPS
    out = set()
    for key in ("A", "B"):
        for r in partners[key]:
            if r["resname"] not in STANDARD_AA:
                continue
            h = _heavy_coords(r)
            if len(h) == 0:
                continue
            d2 = ((h[:, None, :] - O[None, :, :]) ** 2).sum(-1)
            if math.sqrt(float(d2.min())) <= thr:
                out.add(int(r["resid"]))
    return out


def expand_resids_chemistry(partners, resids):
    """Apply PRO / lone-GLY / disulfide pull-in to a residue-number set."""
    aa_a = [r for r in partners["A"] if r["resname"] in STANDARD_AA]
    aa_b = [r for r in partners["B"] if r["resname"] in STANDARD_AA]
    sel = {id(r) for r in aa_a + aa_b if int(r["resid"]) in resids}
    _apply_chemistry_rules(aa_a, sel)
    _apply_chemistry_rules(aa_b, sel)
    return {int(r["resid"]) for r in aa_a + aa_b if id(r) in sel}


def _snapshot_protein_res_count(snapshot_pdb):
    """Highest protein residue serial in a cpptraj snapshot (waters/ions excluded)."""
    max_res = 0
    skip = WATER_RESNAMES | {"NA", "CL", "ACE", "NME"}
    with open(snapshot_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname in skip:
                continue
            try:
                max_res = max(max_res, int(line[22:26]))
            except ValueError:
                pass
    return max_res


def assign_waters_nearer(pdb_in, pdb_out):
    """Rewrite chain IDs so each water belongs to the nearer protein partner.

    For every water residue (HOH/WAT/...), the oxygen-to-heavy-atom minimum
    distance is taken over the protein and cap atoms of each partner; the water
    goes to the closer partner and ties stay with A. Only the chain-ID column
    changes, so atom order, coordinates, charges and the water selection itself
    are unchanged. Used by the nearer-partner test of Supporting Information S7.
    """
    lines = Path(pdb_in).read_text().splitlines(keepends=True)
    heavy = {"A": [], "B": []}
    waters = {}
    order = []
    for i, ln in enumerate(lines):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        resname = ln[17:20].strip().upper()
        if resname in WATER_RESNAMES:
            key = ln[21:27]
            if key not in waters:
                waters[key] = {"idx": [], "O": None}
                order.append(key)
            waters[key]["idx"].append(i)
            if el == "O":
                waters[key]["O"] = xyz
            continue
        if el != "H":
            heavy[ln[21] if ln[21] in "AB" else "A"].append(xyz)
    if not heavy["A"] or not heavy["B"]:
        raise ValueError(f"{pdb_in}: need protein atoms on both chain A and chain B")
    ha = np.array(heavy["A"])
    hb = np.array(heavy["B"])
    n_to_b = 0
    for key in order:
        o = waters[key]["O"]
        if o is None:
            o = (float(lines[waters[key]["idx"][0]][30:38]),
                 float(lines[waters[key]["idx"][0]][38:46]),
                 float(lines[waters[key]["idx"][0]][46:54]))
        o = np.asarray(o)
        da = float(np.sqrt(((ha - o) ** 2).sum(1).min()))
        db = float(np.sqrt(((hb - o) ** 2).sum(1).min()))
        if db < da:
            n_to_b += 1
            for i in waters[key]["idx"]:
                ln = lines[i]
                lines[i] = ln[:21] + "B" + ln[22:]
    Path(pdb_out).write_text("".join(lines))
    return {"n_waters": len(order), "n_waters_b": n_to_b,
            "n_waters_a": len(order) - n_to_b}


def trim_waters(pdb_in, pdb_out, max_waters):
    """Keep only the first ``max_waters`` water residues, in file order.

    The capped models of the paper list waters by bridging rank, so trimming
    to N keeps the N highest-ranked waters.
    """
    lines = Path(pdb_in).read_text().splitlines(keepends=True)
    out, kept, current = [], 0, None
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            resname = ln[17:20].strip().upper()
            key = ln[21:27]
            if resname in WATER_RESNAMES:
                if key != current:
                    current = key
                    if kept >= max_waters:
                        current = None
                        continue
                    kept += 1
                elif current is None:
                    continue
            else:
                current = None
        out.append(ln)
    Path(pdb_out).write_text("".join(out))
    return kept


def resolve_snapshot_chain_split(snapshot, input_pdb):
    """Map the input-PDB chain A length onto a cpptraj snapshot's residue numbering.

    A snapshot can contain fewer protein residues than the input PDB, so the
    chain A residue count of the input is scaled by the ratio of protein residue
    counts (snapshot vs input complex). Returns (n_chain_a, protein_res_count).
    """
    from common import get_first_chain_nres, get_protein_residue_count
    nA_in = get_first_chain_nres(input_pdb)
    prot_in = get_protein_residue_count(input_pdb)
    prot_snap = _snapshot_protein_res_count(snapshot)
    if prot_in <= 0 or prot_snap <= 1:
        return nA_in, prot_in
    nA = int(round(nA_in * prot_snap / prot_in))
    nA = max(1, min(nA, prot_snap - 1))
    return nA, prot_snap


def _residue_terminal_charge(resrec):
    """Formal charge of a real chain terminus retained in the truncated shell.

    RESIDUE_CHARGES covers side-chain ionisation only and assumes the N- and
    C-terminal charges of a whole chain cancel; after truncation only one
    terminus may remain, so its charge is added here. Detected from atom names:
      * at least two of H1/H2/H3: protonated N-terminus, +1 (NH3+ or proline NH2+),
      * OXT: carboxylate C-terminus, -1.
    Caps and waters (whose H1/H2 would otherwise match) return 0.
    """
    if resrec["resname"] in WATER_RESNAMES or resrec["resname"] in CAP_RESNAMES:
        return 0
    names = {a for (a, _xyz, _el) in resrec["atoms"]}
    c = 0
    if len({"H1", "H2", "H3"} & names) >= 2:
        c += 1
    if "OXT" in names:
        c -= 1
    return c


def _partner_charge(selected_resrecs):
    total = 0
    for r in selected_resrecs:
        total += RESIDUE_CHARGES.get(r["resname"], 0)
        total += _residue_terminal_charge(r)
    return total


def truncate(pdb_path, out_pdb, n_chain_a, protein_res_count, cutoff,
             mobile_sidechains=False, keep_waters=True, max_waters=None,
             caps="single", polar_waters=False, interface_mode="distance",
             dasa_resids=None, water_radius=None, water_select=None):
    """Build the capped two-sided interface shell and return a stats dict.

    ``interface_mode``: ``distance`` (heavy-atom cutoff), ``dasa`` (``dasa_resids``
    with a distance fallback for an empty face) or ``dasa_union`` (``dasa_resids``
    used as given, without chemistry pull-in).
    ``caps``: ``ace_nme`` (ACE/NME caps, for the tleap MM/GB path) or ``single``
    (BNC/BCC/BCG/BCB caps, for direct single points).
    ``water_radius``: water-selection distance; defaults to ``cutoff``.
    ``water_select``: ``cutoff`` (waters within ``water_radius`` of the interface)
    or ``closest`` (the ``max_waters`` nearest waters, default 20); defaults to
    ``closest`` for ``dasa_union`` and ``cutoff`` otherwise.
    """
    partners = parse_snapshot(pdb_path, n_chain_a, protein_res_count)
    if interface_mode in ("dasa", "dasa_union"):
        if not dasa_resids:
            raise ValueError(f"interface_mode={interface_mode} requires dasa_resids")
        if interface_mode == "dasa_union":
            sel = select_interface_dasa(partners, dasa_resids)
        else:
            sel = _dasa_with_bilateral_fallback(partners, dasa_resids, cutoff)
        if not sel:
            raise ValueError(f"No dASA interface residues matched in {pdb_path}")
    else:
        sel = select_interface(partners, cutoff)
        if not sel:
            raise ValueError(f"No interface residues within {cutoff} A in {pdb_path}")

    aa_a = [r for r in partners["A"] if r["resname"] in STANDARD_AA]
    aa_b = [r for r in partners["B"] if r["resname"] in STANDARD_AA]
    if interface_mode != "dasa_union":
        _apply_chemistry_rules(aa_a, sel)
        _apply_chemistry_rules(aa_b, sel)

    em = _Emitter()
    # Chain A: partner-A fragments and caps, followed by interfacial waters
    # (waters are assigned to partner A).
    _emit_partner(em, aa_a, sel, "A", mobile_sidechains, caps=caps)
    w_rad = cutoff if water_radius is None else water_radius
    if water_select is None:
        water_select = "closest" if interface_mode == "dasa_union" else "cutoff"
    if not keep_waters:
        waters = []
    elif water_select == "closest":
        nkeep = 20 if max_waters is None else max_waters
        waters = _closest_waters(partners, sel, nkeep, polar_waters=polar_waters)
    else:
        waters = _interfacial_waters(partners, sel, w_rad, max_waters=max_waters,
                                    polar_waters=polar_waters)
    for w in waters:
        em.water(w, "A", mobile_sidechains)
        em.ter()
    # Chain B: partner-B fragments + caps.
    _emit_partner(em, aa_b, sel, "B", mobile_sidechains, caps=caps)
    em.lines.append("END\n")

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text("".join(em.lines))

    sel_a_recs = [r for r in aa_a if id(r) in sel]
    sel_b_recs = [r for r in aa_b if id(r) in sel]
    charge_a = _partner_charge(sel_a_recs)
    charge_b = _partner_charge(sel_b_recs)
    sum_z = 0
    # Nuclear charge per partner: each partner is computed as a separate
    # closed-shell singlet and needs an even electron count of its own. These
    # values are only reported (cuby_jobs uses them to choose per-partner
    # charges); the parity fix below acts on the total.
    sum_z_a = sum_z_b = 0
    for ln in em.lines:
        if ln.startswith("ATOM"):
            el = ln[76:78].strip().upper()
            z = _ATOMIC_Z.get(el, 0)
            sum_z += z
            if ln[21] == "A":
                sum_z_a += z
            else:
                sum_z_b += z
    # Closed-shell singlet: the electron count (sum_z - charge) must be even.
    # If it is odd, the charge of the less positive partner is raised by +1.
    total_charge = charge_a + charge_b
    parity_fix = 0
    if (sum_z - total_charge) % 2 != 0:
        if charge_b <= charge_a:
            charge_b += 1
        else:
            charge_a += 1
        parity_fix = 1
    electrons_even = ((sum_z - (charge_a + charge_b)) % 2 == 0)
    stats = {
        "cutoff": cutoff,
        "n_res_a": len(sel_a_recs),
        "n_res_b": len(sel_b_recs),
        "n_waters": len(waters),
        "n_atoms": sum(1 for ln in em.lines if ln.startswith("ATOM")),
        "charge_a": charge_a,
        "charge_b": charge_b,
        "charge_parity_fix": parity_fix,
        "sum_z": sum_z,
        "sum_z_a": sum_z_a,
        "sum_z_b": sum_z_b,
        "electrons_even": electrons_even,
        "mobile_atoms": len(em.mobile),
        "polar_waters": polar_waters,
        "water_radius": w_rad,
        "water_select": water_select,
        "interface_mode": interface_mode,
        "caps": caps,
        "n_dasa_resids": len(dasa_resids or []),
    }
    # Sidecar JSON with charges and the mobile-atom selection for SQM optimization.
    sidecar = out_pdb.with_suffix(".trunc.json")
    sidecar.write_text(json.dumps({
        "v": 8,  # sidecar format version
        "source": str(pdb_path),
        "cutoff": cutoff,
        "polar_waters": polar_waters,
        "water_radius": w_rad,
        "water_select": water_select,
        "interface_mode": interface_mode,
        "caps": caps,
        "n_dasa_resids": len(dasa_resids or []),
        "n_chain_a": n_chain_a,
        "protein_res_count": protein_res_count,
        "charge_a": stats["charge_a"],
        "charge_b": stats["charge_b"],
        "charge_parity_fix": stats.get("charge_parity_fix", 0),
        "sum_z": stats["sum_z"],
        "sum_z_a": stats["sum_z_a"],
        "sum_z_b": stats["sum_z_b"],
        "electrons_even": stats["electrons_even"],
        "n_res_a": stats["n_res_a"],
        "n_res_b": stats["n_res_b"],
        "n_waters": stats["n_waters"],
        "n_atoms": stats["n_atoms"],
        "mobile_atoms": em.mobile,
    }))
    return stats


def mobile_atom_indices(capped_pdb):
    """1-based serials of relaxable atoms (side chains of standard residues +
    waters; backbone and caps frozen) for an already-written capped PDB."""
    sidecar = Path(capped_pdb).with_suffix(".trunc.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text()).get("mobile_atoms", [])
    mobile = []
    idx = 0
    for line in Path(capped_pdb).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        idx += 1
        resname = line[17:20].strip().upper()
        atom = line[12:16].strip().upper()
        if resname in CAP_RESNAMES:
            continue
        if resname in WATER_RESNAMES or resname == "HOH":
            mobile.append(idx)
        elif resname in STANDARD_AA and atom not in BACKBONE_ATOMS:
            mobile.append(idx)
    return mobile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True, help="dry_f*/wet_f* snapshot PDB")
    ap.add_argument("--out", required=True, help="output capped interface PDB")
    ap.add_argument("--cutoff", type=float, default=8.0, help="heavy-atom cutoff (A)")
    ap.add_argument("--input-pdb", help="original input PDB to derive the A/B split")
    ap.add_argument("--n-chain-a", type=int, help="residues in partner A")
    ap.add_argument("--protein-res-count", type=int, help="total protein residues")
    ap.add_argument("--dasa-resids",
                    help="select these interface residues instead of the distance cutoff: "
                         "a cpptraj mask such as :37,38,40-45, or a JSON file with a "
                         "'resids' list as written by interface_dasa.py")
    ap.add_argument("--max-waters", type=int, help="keep at most this many interfacial waters")
    ap.add_argument("--mobile-sidechains", action="store_true",
                    help="tag side-chain + water atoms as mobile (for SQM optimize)")
    ap.add_argument("--no-waters", action="store_true", help="drop waters even if present")
    ap.add_argument("--polar-waters", action="store_true",
                    help="anchor interfacial waters on POLAR interface residues only")
    args = ap.parse_args()

    dasa_resids = None
    if args.dasa_resids:
        mask = args.dasa_resids
        if Path(mask).is_file():
            dasa_resids = set(json.loads(Path(mask).read_text())["resids"])
        else:
            dasa_resids = set()
            for part in mask.lstrip(":").split(","):
                lo, _, hi = part.strip().partition("-")
                if lo:
                    dasa_resids.update(range(int(lo), int(hi or lo) + 1))
        if not dasa_resids:
            ap.error(f"--dasa-resids {mask!r} selects no residues")

    if args.input_pdb:
        n_chain_a, protein_res_count = resolve_snapshot_chain_split(args.pdb, args.input_pdb)
    else:
        if args.n_chain_a is None or args.protein_res_count is None:
            ap.error("provide --input-pdb OR both --n-chain-a and --protein-res-count")
        n_chain_a, protein_res_count = args.n_chain_a, args.protein_res_count

    stats = truncate(
        args.pdb, args.out, n_chain_a, protein_res_count, args.cutoff,
        mobile_sidechains=args.mobile_sidechains, keep_waters=not args.no_waters,
        max_waters=args.max_waters, polar_waters=args.polar_waters,
        interface_mode="dasa" if dasa_resids else "distance", dasa_resids=dasa_resids,
    )
    print(f"A/B split: n_chain_a={n_chain_a} protein_res_count={protein_res_count}")
    print(f"Wrote {args.out}: {stats}")


if __name__ == "__main__":
    main()
