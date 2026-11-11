#!/usr/bin/env python3
"""Frame-union ("locked") interface residue set for truncated shells.

Per full-solvent snapshot: dASA interface residues, the N waters closest to them,
and residues within ``grow_radius`` of those waters. The union over frames,
extended by PRO/GLY/disulfide rules, is cached as JSON under
interface_masks/dasa_union/ and used by truncate_interface.truncate
(interface_mode="dasa_union").
"""
from __future__ import annotations

import json
from pathlib import Path

import interface_dasa as ID
import truncate_interface as TI

from paths import WORK

REPO = WORK
CACHE_DIR = REPO / "interface_masks" / "dasa_union"
DEFAULT_GROW = 4.0
DEFAULT_NWAT = 20
DEFAULT_DASA = 0.5


def _sel_from_resids(partners, resids):
    return {id(r) for key in ("A", "B") for r in partners[key]
            if r["resname"] in TI.STANDARD_AA and int(r["resid"]) in resids}


def closest_waters_to_resids(partners, resids, max_waters, polar=True):
    sel = _sel_from_resids(partners, resids)
    return TI._closest_waters(partners, sel, max_waters, polar_waters=polar)


def frame_contributions(full_pdb, n_chain_a, protein_res_count,
                        dasa_cutoff=DEFAULT_DASA, nwat=DEFAULT_NWAT,
                        grow_radius=DEFAULT_GROW, polar_waters=True):
    """dASA + water-contact residue sets for one full-solvent snapshot."""
    partners = TI.parse_snapshot(full_pdb, n_chain_a, protein_res_count)
    dasa = ID.dasa_on_snapshot(full_pdb, n_chain_a, protein_res_count,
                               dasa_cutoff=dasa_cutoff, polar=False)
    waters = closest_waters_to_resids(partners, dasa, nwat, polar=polar_waters)
    water_res = TI.residues_near_waters(partners, waters, radius=grow_radius)
    return {
        "dasa": sorted(dasa),
        "water_contact": sorted(water_res),
        "union": sorted(dasa | water_res),
        "n_waters_used": len(waters),
    }


def lock_resids(per_frame, partners_for_chem):
    locked = set()
    for fr in per_frame:
        locked.update(fr["dasa"])
        locked.update(fr["water_contact"])
    # Chemistry rules are applied on one snapshot (PRO/GLY/disulfide topology is
    # identical across frames), iterating to a fixed point.
    for _ in range(3):
        expanded = TI.expand_resids_chemistry(partners_for_chem, locked)
        if expanded == locked:
            break
        locked = expanded
    return locked


def cache_path(system, rep, dasa_cutoff=DEFAULT_DASA, grow_radius=DEFAULT_GROW,
               nwat=DEFAULT_NWAT, polar_waters=True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    polar = "polar" if polar_waters else "all"
    return CACHE_DIR / (
        f"{system}_{rep}_dasa{dasa_cutoff:g}_g{grow_radius:g}_w{nwat}_{polar}.json"
    )


def build_locked_mask(system, frames_full, n_chain_a, protein_res_count,
                      dasa_cutoff=DEFAULT_DASA, nwat=DEFAULT_NWAT,
                      grow_radius=DEFAULT_GROW, polar_waters=True,
                      crystal_dasa=None, rep="r1", refresh=False):
    """Build (or load) the locked residue set for one (system, replica)."""
    cfile = cache_path(system, rep, dasa_cutoff, grow_radius, nwat, polar_waters)
    if cfile.exists() and not refresh:
        return json.loads(cfile.read_text())

    per_frame = []
    for i, pdb in enumerate(frames_full, start=1):
        rec = frame_contributions(pdb, n_chain_a, protein_res_count,
                                  dasa_cutoff=dasa_cutoff, nwat=nwat,
                                  grow_radius=grow_radius,
                                  polar_waters=polar_waters)
        rec["frame"] = i
        rec["pdb"] = str(pdb)
        per_frame.append(rec)

    chem_partners = TI.parse_snapshot(frames_full[0], n_chain_a, protein_res_count)
    locked = lock_resids(per_frame, chem_partners)
    dasa_union = set()
    water_union = set()
    for fr in per_frame:
        dasa_union.update(fr["dasa"])
        water_union.update(fr["water_contact"])

    crystal = set(crystal_dasa or [])
    payload = {
        "system": system,
        "rep": rep,
        "dasa_cutoff": dasa_cutoff,
        "grow_radius": grow_radius,
        "nwat": nwat,
        "polar_waters": polar_waters,
        "n_chain_a": n_chain_a,
        "protein_res_count": protein_res_count,
        "locked_resids": sorted(locked),
        "dasa_union": sorted(dasa_union),
        "water_contact_union": sorted(water_union),
        "water_only_extras": sorted(water_union - dasa_union),
        "crystal_dasa": sorted(crystal),
        "jaccard_crystal_vs_dasa_union": (
            (len(crystal & dasa_union) / len(crystal | dasa_union))
            if (crystal or dasa_union) else None
        ),
        "per_frame": per_frame,
    }
    cfile.write_text(json.dumps(payload, indent=2))
    return payload


def get_locked_resids(system, rep, in_base, n_chain_a, protein_res_count,
                      frames, ff="ff14sb", dasa_cutoff=DEFAULT_DASA,
                      nwat=DEFAULT_NWAT, grow_radius=DEFAULT_GROW,
                      polar_waters=True, refresh=False):
    """Resolve full_f*.pdb frames and return the locked residue set."""
    in_base = Path(in_base)
    pdbs = []
    for f in frames:
        p = in_base / system / ff / rep / f"full_f{f}.pdb"
        if not p.exists():
            raise FileNotFoundError(
                f"missing full-solvent snapshot {p}"
            )
        pdbs.append(p)
    crystal = ID.get_mask(system, dasa_cutoff=dasa_cutoff, polar=False)
    rec = build_locked_mask(
        system, pdbs, n_chain_a, protein_res_count,
        dasa_cutoff=dasa_cutoff, nwat=nwat, grow_radius=grow_radius,
        polar_waters=polar_waters, crystal_dasa=crystal, rep=rep,
        refresh=refresh,
    )
    return set(rec["locked_resids"])
