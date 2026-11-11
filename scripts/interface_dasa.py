#!/usr/bin/env python3
"""Buried-SASA (dASA) interface residue masks from PyMOL InterfaceResidues.

Follows the Nwat-MMGBSA interface definition (Maffucci & Contini, 2016): dASA
cutoff 0.5 A^2 on complex_<ID>_leap.pdb or on MD snapshots, with an optional
POLAR filter for water-anchoring residues. Masks are cached as JSON under
interface_masks/.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from paths import BENCHMARK, INTERFACE_PML, MASK_DIR, PYMOL_EXE, WORK

REPO = WORK
ART = BENCHMARK
COMPLEXES = ART / "complexes"
CACHE_DIR = MASK_DIR
PYMOL = PYMOL_EXE

POLAR_RESNAMES = {
    "ARG", "ASH", "ASP", "GLH", "GLN", "GLU", "HID", "HIE", "HIP",
    "LYN", "LYS", "SER", "THR", "TRP", "TYR",
}


def _parse_mask_resids(mask: str) -> set[int]:
    """Parse a cpptraj mask such as ':37,38,40' into {37, 38, 40}."""
    if not mask or mask == ":":
        return set()
    body = mask.lstrip(":")
    out = set()
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def interface_mask_pymol(system: str, workdir: Path, dasa_cutoff: float = 0.5,
                         polar: bool = False) -> str | None:
    """Run PyMOL InterfaceResidues; return cpptraj mask ':r1,r2,...'."""
    workdir.mkdir(parents=True, exist_ok=True)
    leap = COMPLEXES / f"complex_{system}_leap.pdb"
    if not leap.exists():
        return None
    pml = workdir / "interface_run.pml"
    pml.write_text(
        f"run {INTERFACE_PML}\n"
        f"load {leap}\n"
        f'myInterfaceResidues = interfaceResidues("complex_{system}_leap", '
        f'cA="c. A", cB="c. B", cutoff={dasa_cutoff}, selName="interface")\n'
        f"save {workdir}/{system}_interface.pdb, interface\nquit\n")
    subprocess.run([PYMOL, "-cq", str(pml)], cwd=workdir, capture_output=True, text=True)
    ipdb = workdir / f"{system}_interface.pdb"
    if not ipdb.exists():
        return None
    resids = []
    for ln in ipdb.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            r = ln[22:26].strip()
            if polar and ln[17:20].strip() not in POLAR_RESNAMES:
                continue
            if r and r not in resids:
                resids.append(r)
    return ":" + ",".join(resids) if resids else None


def get_mask(system: str, dasa_cutoff: float = 0.5, polar: bool = False,
             refresh: bool = False) -> set[int]:
    """Cached dASA residue set (global residue numbers)."""
    tag = f"dasa{dasa_cutoff:g}" + ("_polar" if polar else "")
    cache = CACHE_DIR / tag
    cache.mkdir(parents=True, exist_ok=True)
    cfile = cache / f"{system}.json"
    if cfile.exists() and not refresh:
        return set(json.loads(cfile.read_text())["resids"])
    mask = interface_mask_pymol(system, cache / "_work" / system,
                                dasa_cutoff=dasa_cutoff, polar=polar)
    resids = sorted(_parse_mask_resids(mask or ""))
    cfile.write_text(json.dumps({
        "system": system, "dasa_cutoff": dasa_cutoff, "polar": polar,
        "mask": mask, "resids": resids,
    }))
    return set(resids)


_WATER_RESNAMES = {"HOH", "WAT", "TP3", "TIP3", "TIP4", "SOL", "OPC", "T3P"}


def write_snapshot_with_chains(pdb_path, out_path, n_chain_a, protein_res_count):
    """Rewrite a cpptraj snapshot so residue 1..n_chain_a is chain A and the
    rest of the protein is chain B. Waters/ions are dropped (dASA is protein-only)."""

    lines = []
    pdb_path, out_path = Path(pdb_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for raw in pdb_path.read_text(errors="ignore").splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        try:
            resid = int(raw[22:26])
        except ValueError:
            continue
        resname = raw[17:20].strip().upper()
        if resname in _WATER_RESNAMES or resid > protein_res_count:
            continue
        chain = "A" if resid <= n_chain_a else "B"
        line = raw.ljust(80)
        lines.append(line[:21] + chain + line[22:])
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def dasa_on_snapshot(pdb_path, n_chain_a, protein_res_count, dasa_cutoff=0.5,
                     polar=False, workdir=None, refresh=False):
    """PyMOL InterfaceResidues on an MD snapshot, using the residue-number A/B split.

    Returns a set of global residue numbers. Cached under
    ``interface_masks/dasa_frame/`` keyed by snapshot path + cutoff.
    """
    pdb_path = Path(pdb_path).resolve()
    tag = f"dasa{dasa_cutoff:g}" + ("_polar" if polar else "")
    cache_dir = CACHE_DIR / "dasa_frame" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{pdb_path.stem}_{n_chain_a}_{protein_res_count}"
    # Prefix with the (truncated) parent path so identically named snapshots from
    # different replicas do not collide.
    parent = pdb_path.parent.as_posix().replace("/", "_")[-80:]
    cfile = cache_dir / f"{parent}_{key}.json"
    if cfile.exists() and not refresh:
        return set(json.loads(cfile.read_text())["resids"])

    work = Path(workdir) if workdir else (cache_dir / "_work" / key)
    work.mkdir(parents=True, exist_ok=True)
    chained = work / "snapshot_chains.pdb"
    write_snapshot_with_chains(pdb_path, chained, n_chain_a, protein_res_count)
    obj = chained.stem
    pml = work / "interface_run.pml"
    ipdb = work / "interface.pdb"
    pml.write_text(
        f"run {INTERFACE_PML}\n"
        f"load {chained}, {obj}\n"
        f'myInterfaceResidues = interfaceResidues("{obj}", '
        f'cA="c. A", cB="c. B", cutoff={dasa_cutoff}, selName="interface")\n'
        f"save {ipdb}, interface\nquit\n")
    subprocess.run([PYMOL, "-cq", str(pml)], cwd=work, capture_output=True, text=True)
    resids = []
    if ipdb.exists():
        seen = set()
        for ln in ipdb.read_text().splitlines():
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            r = ln[22:26].strip()
            if not r or r in seen:
                continue
            if polar and ln[17:20].strip() not in POLAR_RESNAMES:
                continue
            seen.add(r)
            resids.append(int(r))
    cfile.write_text(json.dumps({
        "source": str(pdb_path), "dasa_cutoff": dasa_cutoff, "polar": polar,
        "n_chain_a": n_chain_a, "protein_res_count": protein_res_count,
        "resids": resids,
    }))
    return set(resids)


def polar_mask_for_waters(system: str, dasa_cutoff: float = 0.5) -> str | None:
    """cpptraj selection string for POLAR dASA interface (water anchoring)."""
    cache = CACHE_DIR / f"dasa{dasa_cutoff:g}_polar" / f"{system}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("mask")
    get_mask(system, dasa_cutoff=dasa_cutoff, polar=True)
    if cache.exists():
        return json.loads(cache.read_text()).get("mask")
    return None
