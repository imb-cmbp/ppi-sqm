#!/usr/bin/env python3
"""MM/GB (AMBER tleap/sander) relaxation of a capped interface shell.

Builds ff14SB topologies, runs restrained GB minimization or short GB MD, and maps
the relaxed coordinates back onto the capped template. Used by score_mopac.py
(phase ``mm_sp``) before the SQM interaction energy; follows the SQM//MM protocol
of Yurenko and Lepšík.
"""
from __future__ import annotations

import fcntl
import math
import re
from pathlib import Path

from amber_setup import amber_bin, amber_env
from common import safe_run

from paths import CAP_PARAMS

FF = CAP_PARAMS

_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "NA": 11, "CL": 17,
      "MG": 12, "ZN": 30, "CA": 20, "FE": 26, "K": 19}

# Chain-ID pools for partners A and B. Each TER-separated fragment gets its own
# chain ID so that tleap treats it as a separate chain; IDs must be unique within
# a shell because (chain, resseq) identifies residues downstream.
_A_POOL = list("ACEFGIKMOQSUWY") + list("acefgikmoqsuwy") + list("02468")
_B_POOL = list("BDHJLNPRTVXZ") + list("bdhjlnprtvxz") + list("13579")
_WATER = {"HOH", "WAT", "TP3", "TIP3", "TIP4", "SOL", "OPC", "T3P"}
_PARMED_LOCK = Path("/tmp/ppi_sqm_parmed.lock")


def _run_parmed(cmd, workdir, env):
    """Run ParmEd under a file lock; its CLI startup is unreliable under concurrent launches."""
    with open(_PARMED_LOCK, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        return safe_run(cmd, workdir, env)


def _elem(line):
    e = line[76:78].strip().upper()
    if e:
        return e
    nm = line[12:16].strip()
    return "".join(c for c in nm if c.isalpha())[:2].upper()


def disulfide_bonds(pdb_path):
    """(i, j) 1-based residue indices of SG-SG pairs < 2.5 A.

    Indices must match tleap's numbering, which is sequential in file order, so
    residues are counted on every change of residue key rather than by unique
    key; counting unique keys would undercount when two fragments share a
    (chain, resseq) label.
    """
    sg = []
    ridx = 0
    prev_key = None
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:27])
        if key != prev_key:
            prev_key = key
            ridx += 1
        if line[12:16].strip() == "SG" and line[17:20].strip() in ("CYX", "CYS"):
            sg.append((ridx, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    bonds, used = [], set()
    for a in range(len(sg)):
        if sg[a][0] in used:
            continue
        for b in range(a + 1, len(sg)):
            if sg[b][0] in used:
                continue
            if math.dist(sg[a][1], sg[b][1]) < 2.5:
                bonds.append((sg[a][0], sg[b][0]))
                used.add(sg[a][0])
                used.add(sg[b][0])
                break
    return bonds


def chain_a_nres(pdb_path):
    """Residue count on chain A (protein + waters) and total residues."""
    a, b = [], []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM"):
            (a if line[21] == "A" else b).append(line[22:27])
    ka = len(dict.fromkeys(a))
    return ka, ka + len(dict.fromkeys(b))


def chain_sumz(pdb_path):
    """Sum of atomic numbers per capped-PDB chain (A vs B)."""
    za = zb = 0
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        z = _Z.get(_elem(line), 0)
        if line[21] == "A":
            za += z
        else:
            zb += z
    return za, zb


def _fragment_partner(lines):
    """Partner A if any protein atom is chain A or any water; else B."""
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        resn = ln[17:20].strip()
        if resn in _WATER:
            return "A"
        if ln[21] == "A":
            return "A"
    return "B"


_CAP = {"BNC", "BCC", "BCG", "BCB"}


def _strip_uncapped_nt_h(frag):
    """Drop the backbone amide H at uncapped fragment starts (tleap N-terminal templates use H1/H2/H3)."""
    out, cur_res, first_aa = [], None, None
    for ln in frag:
        if not ln.startswith("ATOM"):
            out.append(ln)
            continue
        reskey = ln[17:27]
        resn = ln[17:20].strip()
        if reskey != cur_res:
            cur_res = reskey
            if first_aa is None and resn not in _CAP and resn not in _WATER:
                first_aa = reskey
        if (first_aa == reskey and ln[12:16].strip() == "H"
                and resn not in _CAP):
            continue
        out.append(ln)
    return out


def _coalesce_fragments(frags):
    """Merge consecutive water-only TER fragments (one HOH per TER in capped PDB)."""
    out, water_buf = [], []
    for frag in frags:
        if not frag:
            continue
        resnames = {ln[17:20].strip() for ln in frag}
        if resnames <= _WATER:
            water_buf.extend(frag)
            continue
        if water_buf:
            if out and _fragment_partner(out[-1]) == "A":
                out[-1].extend(water_buf)
            else:
                out.append(water_buf)
            water_buf = []
        out.append(frag)
    if water_buf:
        if out and _fragment_partner(out[-1]) == "A":
            out[-1].extend(water_buf)
        else:
            out.append(water_buf)
    return out


def prepare_tleap_pdb(capped_pdb, out_pdb):
    """Split TER-separated fragments onto unique chain IDs for tleap.

    Multi-fragment capped shells reuse residue numbers on one chain; without
    separate chains tleap treats each new fragment as an N-terminus and fails.
    """
    raw = Path(capped_pdb).read_text().splitlines()
    frags, cur = [], []
    for ln in raw:
        if ln.startswith("TER"):
            if cur:
                frags.append(cur)
                cur = []
            continue
        if ln.startswith("ATOM"):
            cur.append(ln)
    if cur:
        frags.append(cur)
    frags = _coalesce_fragments(frags)

    ai = bi = 0
    out = []
    for frag in frags:
        if not frag:
            continue
        frag = _strip_uncapped_nt_h(frag)
        partner = _fragment_partner(frag)
        if partner == "A":
            chain = _A_POOL[ai % len(_A_POOL)]
            ai += 1
        else:
            chain = _B_POOL[bi % len(_B_POOL)]
            bi += 1
        resi = 0
        cur_res = None
        for ln in frag:
            reskey = ln[17:27]
            if reskey != cur_res:
                cur_res = reskey
                resi += 1
            out.append(ln[:21] + chain + f"{resi:4d}" + ln[26:] + "\n")
        out.append("TER\n")
    Path(out_pdb).write_text("".join(out))


def tleap_partner_a_nres(tleap_pdb):
    """Return (residues on partner-A chains, total residues) for a tleap PDB.

    Counted in file order, as in disulfide_bonds: unique-key counting would
    merge residues when two fragments share a (chain, resseq) label and shift
    the A/B split used for the interaction energy.
    """
    a_chains = set(_A_POOL)
    ka = total = 0
    prev_key = None
    for ln in Path(tleap_pdb).read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        key = (ln[21], ln[22:26].strip(), ln[17:20].strip())
        if key == prev_key:
            continue
        prev_key = key
        total += 1
        if ln[21] in a_chains:
            ka += 1
    return ka, total


def count_close_contacts(pdb_path, cutoff=0.5):
    """Number of heavy-atom pairs closer than ``cutoff`` A (10**9 if any coordinate is NaN).

    A nonzero count indicates a broken geometry. A cell grid of side ``cutoff``
    keeps the cost linear in atom count.
    """
    pts = []
    for ln in Path(pdb_path).read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                return 10**9
            name = ln[12:16].strip()
            if name.startswith("H") or (len(name) > 1 and name[0].isdigit() and name[1] == "H"):
                continue
            pts.append((x, y, z))
    grid = {}
    for i, p in enumerate(pts):
        grid.setdefault((int(p[0] // cutoff), int(p[1] // cutoff),
                         int(p[2] // cutoff)), []).append(i)
    n = 0
    for (cx, cy, cz), idxs in grid.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh += grid.get((cx + dx, cy + dy, cz + dz), ())
        for i in idxs:
            for j in neigh:
                if j > i and math.dist(pts[i], pts[j]) < cutoff:
                    n += 1
    return n


def transfer_coords(min_pdb, template_pdb, out_pdb):
    """Copy MM-relaxed coordinates onto the capped template PDB.

    The template keeps its chains, atom names and hydrogens. tleap/cpptraj may
    reorder residues (e.g. waters), so residues are matched by heavy-atom name
    signature; among candidates with the same signature, the one whose centroid
    is closest to the template residue is used, since minimization moves a
    residue by well under 1 A. Residues without a match within ``max_shift``
    keep their template coordinates.
    """

    def residue_blocks(path):
        blocks, cur, atoms = [], None, {}
        for ln in Path(path).read_text().splitlines():
            if not ln.startswith("ATOM"):
                continue
            # Chain is part of the identity: fragments restart numbering, so
            # (resseq, resname) alone merges residues from different fragments.
            key = (ln[21], ln[22:26].strip(), ln[17:20].strip())
            if key != cur:
                if cur is not None:
                    blocks.append(atoms)
                cur = key
                atoms = {}
            atoms[ln[12:16].strip()] = ln[30:54]
        if cur is not None:
            blocks.append(atoms)
        return blocks

    def _is_h(name):
        return name.startswith("H") or (len(name) > 1 and name[0].isdigit()
                                        and name[1] == "H")

    def sig(atoms):
        # Heavy atoms only: tleap adds missing hydrogens, so hydrogen sets can
        # differ between the template and the minimized structure.
        return tuple(sorted(n for n in atoms if not _is_h(n)))

    def centroid(atoms):
        heavy = [c for n, c in atoms.items() if not _is_h(n)] or list(atoms.values())
        if not heavy:
            return None
        sx = sy = sz = 0.0
        for c in heavy:
            sx += float(c[0:8])
            sy += float(c[8:16])
            sz += float(c[16:24])
        n = len(heavy)
        return (sx / n, sy / n, sz / n)

    # Maximum centroid shift (A) for a match; beyond it the template
    # coordinates are kept.
    max_shift = 3.0

    opt_blocks = residue_blocks(min_pdb)
    opt_sigs = [sig(b) for b in opt_blocks]
    opt_cents = [centroid(b) for b in opt_blocks]
    used = [False] * len(opt_blocks)
    mapped = []
    for tpl_atoms in residue_blocks(template_pdb):
        s = sig(tpl_atoms)
        tc = centroid(tpl_atoms)
        pick, best = None, None
        for j, ob in enumerate(opt_blocks):
            if used[j] or opt_sigs[j] != s:
                continue
            oc = opt_cents[j]
            d = (math.dist(tc, oc) if tc is not None and oc is not None
                 else float("inf"))
            if best is None or d < best:
                pick, best = j, d
        if pick is None or best > max_shift:
            mapped.append(tpl_atoms)
        else:
            used[pick] = True
            mapped.append(opt_blocks[pick])

    out = []
    cur = None
    block_i = -1
    atoms_map = {}
    for ln in Path(template_pdb).read_text().splitlines():
        if not ln.startswith("ATOM"):
            out.append(ln)
            continue
        key = (ln[22:26].strip(), ln[17:20].strip())
        if key != cur:
            cur = key
            block_i += 1
            atoms_map = mapped[block_i] if block_i < len(mapped) else {}
        name = ln[12:16].strip()
        coords = atoms_map.get(name)
        if coords is not None:
            ln = ln[:30] + coords + ln[54:]
        out.append(ln)
    Path(out_pdb).write_text("\n".join(out) + "\n")


def net_charge(parmed, prmtop, mask, workdir, env):
    (workdir / "nc.in").write_text(f"parm {prmtop}\nnetCharge {mask}\nquit\n")
    out, _, _ = _run_parmed(f"{parmed} -O -i nc.in", workdir, env)
    m = re.search(r"is\s+([-\d.]+)", out)
    return int(round(float(m.group(1)))) if m else None


def _tleap_ok(workdir):
    prmtop = workdir / "frag.prmtop"
    if prmtop.exists() and prmtop.stat().st_size > 100:
        return True
    leap = (workdir / "leap.log").read_text(errors="ignore")
    return "saveamberparm" in leap and "Fatal Error" not in leap and prmtop.exists()


def _build_frag_gb(capped_pdb, workdir, mopac_dir, radii="bondi"):
    """Build the tleap topology and set GB radii with ParmEd.

    Returns (env, bins, frag, status); the first three are None on failure.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    capped_pdb = Path(capped_pdb)
    env, _ = amber_env(extra_path=mopac_dir)
    bins = {k: amber_bin(k) for k in ("tleap", "parmed", "sander", "cpptraj")}

    frag = workdir / "frag.pdb"
    prepare_tleap_pdb(capped_pdb, frag)
    bonds = disulfide_bonds(frag)
    bond_lines = "\n".join(f"bond frag.{i}.SG frag.{j}.SG" for i, j in bonds)
    # ACE/NME are native to leaprc.protein.ff14SB. BNC/BCC/BCG/BCB are loaded
    # so the same recipe also builds single-cap shells.
    (workdir / "tleap.in").write_text(
        f"source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
        f"loadamberprep {FF}/BNC.in\nloadamberprep {FF}/BCC.in\n"
        f"loadamberprep {FF}/BCG.in\nloadamberprep {FF}/BCB.in\n"
        f"loadamberparams {FF}/BNC.frcmod\nloadamberparams {FF}/BCC.frcmod\n"
        f"loadamberparams {FF}/BCB.frcmod\nfrag = loadpdb frag.pdb\n{bond_lines}\n"
        f"saveamberparm frag frag.prmtop frag.inpcrd\nquit\n")
    safe_run(f"{bins['tleap']} -f tleap.in", workdir, env)
    if not _tleap_ok(workdir):
        return None, None, None, "tleap_fail"

    radii_cmd = "bondi" if radii.lower() == "bondi" else "mbondi3"
    (workdir / "radii.in").write_text(
        f"parm frag.prmtop\nchangeRadii {radii_cmd}\noutparm frag_gb.prmtop\nquit\n")
    parmed_out, parmed_err, parmed_code = _run_parmed(
        f"{bins['parmed']} -O -i radii.in", workdir, env)
    (workdir / "parmed.stdout").write_text(parmed_out)
    (workdir / "parmed.stderr").write_text(parmed_err)
    (workdir / "parmed.exit").write_text(str(parmed_code))
    if not (workdir / "frag_gb.prmtop").exists():
        return None, None, None, "parmed_fail"
    return env, bins, frag, "ok"


def _finish_relaxed(capped_pdb, workdir, env, bins, frag, rst_name, pdb_name):
    """Convert the restart to PDB, get partner charges, and transfer coordinates onto the template."""
    (workdir / "topdb.in").write_text(
        f"parm frag_gb.prmtop\ntrajin {rst_name}\ntrajout {pdb_name}\nrun\nquit\n")
    safe_run(f"{bins['cpptraj']} -i topdb.in", workdir, env)
    if not (workdir / pdb_name).exists():
        return None, None, None, None, "cpptraj_fail"

    ka, n = tleap_partner_a_nres(frag)
    qa = net_charge(bins["parmed"], "frag_gb.prmtop", f":1-{ka}", workdir, env)
    qb = net_charge(bins["parmed"], "frag_gb.prmtop", f":{ka+1}-{n}", workdir, env)
    if qa is None or qb is None:
        return None, None, None, None, "charge_fail"

    za, zb = chain_sumz(capped_pdb)
    geom = workdir / "geom.pdb"
    transfer_coords(workdir / pdb_name, capped_pdb, geom)
    n_clash = count_close_contacts(geom)
    if n_clash:
        return None, None, None, None, f"transfer_clash:{n_clash}"
    return geom, qa, qb, za + zb, "ok"


def run_mm_gb_min(capped_pdb, workdir, mopac_dir, maxcyc=1000, ncyc=300,
                  igb=7, radii="bondi"):
    """Restrained MM/GB minimization. Returns (geom_pdb, qa, qb, sum_z, status).

    ``radii``: ``bondi`` or ``mbondi3``.  ``igb``: 7 or 8 (GB model).
    """
    workdir = Path(workdir)
    capped_pdb = Path(capped_pdb)
    env, bins, frag, status = _build_frag_gb(capped_pdb, workdir, mopac_dir, radii)
    if status != "ok":
        return None, None, None, None, status

    (workdir / "min.in").write_text(
        "MM/GB restrained min\n &cntrl\n"
        f"  imin=1,maxcyc={maxcyc},ncyc={ncyc},ntb=0,igb={igb},"
        "cut=999.0,saltcon=0.15,ntpr=500,\n"
        "  ntr=1,restraint_wt=10.0,restraintmask='@CA,C,N,O',\n /\n")
    safe_run(f"{bins['sander']} -O -i min.in -o min.out -p frag_gb.prmtop "
             f"-c frag.inpcrd -r min.rst7 -ref frag.inpcrd", workdir, env,
             err_file="min.err")
    if not (workdir / "min.rst7").exists():
        return None, None, None, None, "min_fail"
    return _finish_relaxed(capped_pdb, workdir, env, bins, frag, "min.rst7", "min.pdb")


def run_mm_gb_md(capped_pdb, workdir, mopac_dir, nstlim=5000, dt=0.002,
                 igb=8, radii="mbondi3"):
    """Brief clash-relieving min, then short restrained GB MD.

    Caps and CA stay restrained so the fragment cannot unfold; side chains and
    interfacial waters move. Last-frame coords are mapped back onto the capped
    template for the SQM single point.

    Returns (geom_pdb, qa, qb, sum_z, status).
    """
    workdir = Path(workdir)
    capped_pdb = Path(capped_pdb)
    env, bins, frag, status = _build_frag_gb(capped_pdb, workdir, mopac_dir, radii)
    if status != "ok":
        return None, None, None, None, status

    # 200 steepest-descent steps to relieve truncation clashes before MD.
    (workdir / "min.in").write_text(
        "MM/GB clash-relieve min\n &cntrl\n"
        f"  imin=1,maxcyc=200,ncyc=200,ntb=0,igb={igb},"
        "cut=999.0,saltcon=0.15,ntpr=50,\n"
        "  ntr=1,restraint_wt=10.0,restraintmask=':ACE,NME | @CA',\n /\n")
    safe_run(f"{bins['sander']} -O -i min.in -o min.out -p frag_gb.prmtop "
             f"-c frag.inpcrd -r min.rst7 -ref frag.inpcrd", workdir, env,
             err_file="min.err")
    if not (workdir / "min.rst7").exists():
        return None, None, None, None, "min_fail"

    (workdir / "md.in").write_text(
        "MM/GB restrained Langevin\n &cntrl\n"
        f"  imin=0,irest=0,ntx=1,nstlim={nstlim},dt={dt},\n"
        f"  ntb=0,igb={igb},cut=999.0,saltcon=0.15,\n"
        "  ntt=3,gamma_ln=1.0,temp0=300.0,tempi=300.0,\n"
        "  ntc=2,ntf=2,ntpr=500,ntwx=0,ntwr=0,\n"
        "  ntr=1,restraint_wt=5.0,restraintmask=':ACE,NME | @CA',\n /\n")
    safe_run(f"{bins['sander']} -O -i md.in -o md.out -p frag_gb.prmtop "
             f"-c min.rst7 -r md.rst7 -ref frag.inpcrd", workdir, env,
             err_file="md.err")
    if not (workdir / "md.rst7").exists():
        return None, None, None, None, "md_fail"
    return _finish_relaxed(capped_pdb, workdir, env, bins, frag, "md.rst7", "md.pdb")
