#!/usr/bin/env python3
"""Score capped protein-protein interface models from MD snapshots with cuby4 + MOPAC.

Phases: sp (single point on the capped shell from truncate_interface.py), mm_sp /
md_sp (AMBER GB minimization / short restrained GB MD, then single point), opt
(SQM relaxation of interface side chains and waters, then single point). Per-job
logs are reused on restart; per-system mean/std/n are written to one CSV per cutoff.
Usage: python3 score_mopac.py --systems validation --presets default --phases sp
"""
import argparse
import csv
import fcntl
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import Manager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHAIN_REF, CUBY4_DIR as _CUBY4_DIR, MOPAC_EXE, RUBY_BIN as _RUBY_BIN, WORK  # noqa: E402

REPO_BASE = WORK

from interaction import SQM_PRESETS  # noqa: E402
from common import (  # noqa: E402
    safe_run, get_first_chain_nres, get_protein_residue_count, parse_interaction_energy,
)
import truncate_interface as TI  # noqa: E402
import cuby_jobs as QM  # noqa: E402
import interface_dasa as ID  # noqa: E402
import mm_gb_min as MMGB  # noqa: E402

PDB_INPUT_DIR = CHAIN_REF
RUN_BASE = REPO_BASE / "screening_ff_md_sqm"
OUT_BASE = RUN_BASE / "_qmmm_trunc"

MOPAC_PATH = MOPAC_EXE
MOPAC_DIR = os.path.dirname(MOPAC_PATH)
CUBY4_DIR = str(_CUBY4_DIR)
RUBY_BIN = _RUBY_BIN
SCRIPT_INTERACTION = Path(__file__).resolve().parent / "interaction.py"

PDB_IDS = [
    "1ACB", "1ZHI", "1AVX", "2HLE", "1AY7", "2HRK", "1BVN", "2OOB", "1EMV", "2OUL",
    "1FLE", "2SIC", "1GLA", "2SNI", "1KAC", "2UUY", "1R0R", "3BZD", "1YVB", "3SGB",
]
VALIDATION_SYSTEMS = ["1EMV", "2SNI", "2OOB", "2UUY"]  # strong / mid / weak / weak

# Default preset set: classical PM6 family plus PM6-ML family.
CLASSICAL_PRESETS = [
    "pm6_dhplus_cosmo", "pm6_dhplus_cosmo2", "pm6_d3h4x_cosmo2", "pm6_cosmo", "pm6_cosmo2",
]
GAS_PRESETS = ["pm6_d3h4x", "pm7", "pm7_d3h4x"]
ML_PRESETS = ["pm6_ml", "pm6_ml_cosmo", "pm6_ml_cosmo2"]
# Halogen-free dispersion + H-bond corrections (COSMO only); d3 and h4 isolate
# the two terms of d3h4.
NEWCORR_PRESETS = [
    "pm6_d3h4_cosmo", "pm6_d3h4paren_cosmo", "pm6_d3_cosmo",
    "pm6_h4_cosmo", "pm6_dh2_cosmo",
]
# PM6-D3H4XS (s_bond on chalcogen contacts) and solvated PM7 variants.
EXTRA_PRESETS = ["pm6_d3h4s_cosmo2", "pm7_cosmo2", "pm7_d3h4x_cosmo2"]
# Yurenko et al. (JCIM 2025) protocol: published PM6-D3H4S, the Cuby-side D3H4
# and its variant without charged-group H-bond scaling.
JCIM2025_PRESETS = ["pm6_d3h4s_cosmo", "pm6_d3h4_cosmo_cuby", "pm6_d3h4_cosmo_nofc"]
DEFAULT_PRESETS = CLASSICAL_PRESETS + ML_PRESETS
ALL_PRESETS = list(dict.fromkeys(
    CLASSICAL_PRESETS + NEWCORR_PRESETS + ML_PRESETS + GAS_PRESETS + EXTRA_PRESETS
    + JCIM2025_PRESETS
))

FF = "ff14sb"
REP = "rep1"
N_SNAPSHOTS = 10

# Default concurrency limits for the classical and PM6-ML process pools.
MAX_CPU_WORKERS = 8
DEFAULT_ML_WORKERS = 6
THREADS_PER_SQM = 1
MIN_FREE_GB = 24.0   # a job waits until at least this much RAM (GB) is available

# SQM optimization settings (phase "opt").
OPT_OPTIMIZER = "lbfgs"
OPT_QUALITY = 0.5
OPT_MAXCYCLES = 60


def is_ml_preset(preset):
    return preset.startswith("pm6_ml")


def _avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return 1e9


def _wait_for_memory(min_free_gb, max_wait=900):
    """Wait until at least ``min_free_gb`` GB of RAM is available, at most ``max_wait`` s.

    Each PM6-ML job loads PyTorch and a model, so concurrent starts can exhaust
    memory. Returns the number of seconds waited.
    """
    waited = 0
    while _avail_gb() < min_free_gb and waited < max_wait:
        time.sleep(5)
        waited += 5
    return waited


def sqm_thread_env(threads_per_sqm=1):
    n = str(threads_per_sqm)
    return {
        "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n, "VECLIB_MAXIMUM_THREADS": n, "OMP_DYNAMIC": "FALSE",
        "TORCH_NUM_THREADS": n, "PYTORCH_NUM_THREADS": n,
        "PATH": f"{CUBY4_DIR}:{RUBY_BIN}:{MOPAC_DIR}:{os.environ.get('PATH', '')}",
    }


def _sqm_retryable(err_text):
    """True when another (qa, qb) pair is worth trying."""
    err = err_text.lower()
    return any(x in err for x in (
        "charge is wrong",
        "closed-shell",
        "charge specified",
        "energy not found in mopac",
    ))


def _clean_workdir(workdir, keep=("geom.pdb",)):
    """Drop cuby/MOPAC artifacts so a charge-retry attempt starts clean."""
    if not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        return
    for p in workdir.iterdir():
        if p.name in keep:
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def sqm_log_ok(log_path):
    return log_path.exists() and log_path.read_text(errors="ignore")[:32].startswith("EXIT=0")


def score_from_log(log_path):
    if not sqm_log_ok(log_path):
        return np.nan
    return parse_interaction_energy(log_path.read_text(errors="ignore"))


SIDECAR_VERSION = 8


def _snapshot_protein_res_count(snapshot_pdb):
    """Highest protein residue serial in a cpptraj snapshot (waters/ions excluded)."""
    max_res = 0
    skip = TI.WATER_RESNAMES | {"NA", "CL", "ACE", "NME"}
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


def resolve_snapshot_chain_split(snapshot_dry, input_pdb):
    """Map the input-PDB chain A length onto a cpptraj snapshot's residue numbering.

    A snapshot can contain fewer protein residues than the input PDB, so the
    chain A residue count of the input is scaled by the ratio of protein residue
    counts (dry snapshot vs input complex). Returns (n_chain_a, protein_res_count).
    """
    nA_in = get_first_chain_nres(input_pdb)
    prot_in = get_protein_residue_count(input_pdb)
    prot_snap = _snapshot_protein_res_count(snapshot_dry)
    if prot_in <= 0 or prot_snap <= 1:
        return nA_in, prot_in
    nA = int(round(nA_in * prot_snap / prot_in))
    nA = max(1, min(nA, prot_snap - 1))
    return nA, prot_snap


def _sidecar_ok(capped_pdb, polar_waters=False, interface_mode="distance",
                caps="single", water_select=None, water_radius=None):
    """True if the cached shell sidecar matches the current version and settings."""
    sc = capped_pdb.with_suffix(".trunc.json")
    if not (capped_pdb.exists() and sc.exists()):
        return False
    try:
        m = TI.json.loads(sc.read_text())
    except Exception:  # noqa: BLE001
        return False
    if m.get("v", 1) < SIDECAR_VERSION:
        return False
    if m.get("polar_waters", False) != polar_waters:
        return False
    if m.get("interface_mode", "distance") != interface_mode:
        return False
    if m.get("caps", "single") != caps:
        return False
    if water_select is not None and m.get("water_select") != water_select:
        return False
    if water_radius is not None and m.get("water_radius") != water_radius:
        return False
    mob = m.get("mobile_atoms", [])
    # Mobile indices beyond n_atoms mean TER records were counted as atoms.
    return (not mob) or (max(mob) <= m.get("n_atoms", 0))


def _build_capped(snapshot, capped_pdb, n_chain_a, protein_res_count, cutoff,
                  max_waters=None, polar_waters=False, interface_mode="distance",
                  dasa_resids=None, force=False, caps="single",
                  water_radius=None, water_select=None):
    """Build and cache the capped shell for one (system, cutoff, hydration, frame).

    Mobile side chains are tagged so the sp and opt phases share one geometry.
    The shell is rebuilt when its sidecar is outdated or ``force`` is set.
    """
    if not force and _sidecar_ok(capped_pdb, polar_waters, interface_mode,
                                 caps=caps, water_select=water_select,
                                 water_radius=water_radius):
        return
    TI.truncate(snapshot, capped_pdb, n_chain_a, protein_res_count, cutoff,
                mobile_sidechains=True, keep_waters=True, max_waters=max_waters,
                polar_waters=polar_waters, interface_mode=interface_mode,
                dasa_resids=dasa_resids, caps=caps,
                water_radius=water_radius, water_select=water_select)


def run_one(task):
    (system, cutoff, frame, hydration, preset, phase,
     n_chain_a, protein_res_count, threads, ml_sem, opt_cfg) = task

    iface = opt_cfg.get("interface_mode", "distance")
    snap_name = (
        f"full_f{frame}.pdb" if (iface == "dasa_union" and hydration == "wet")
        else f"{hydration}_f{frame}.pdb"
    )
    snapshot = RUN_BASE / system / FF / opt_cfg.get("rep", REP) / snap_name
    out_dir = OUT_BASE / system / f"c{cutoff:g}" / hydration / f"f{frame}"
    out_dir.mkdir(parents=True, exist_ok=True)
    capped_pdb = out_dir / "capped.pdb"
    log_path = out_dir / f"cuby_{preset}_{phase}.log"

    key = (system, cutoff, hydration, preset, phase, frame)
    force = opt_cfg.get("force", False)
    if not force and sqm_log_ok(log_path):
        return key, score_from_log(log_path), 0.0, "cached"
    if force and log_path.exists():
        log_path.unlink(missing_ok=True)
    if not snapshot.exists():
        return key, np.nan, 0.0, "no_snapshot"

    throttle = ml_sem if (ml_sem is not None and is_ml_preset(preset)) else None
    mm_throttle = opt_cfg.get("mm_sem") if phase in ("mm_sp", "md_sp") else None
    if throttle is not None:
        throttle.acquire()
    if mm_throttle is not None:
        mm_throttle.acquire()

    # Wait for free memory before starting SQM work.
    _wait_for_memory(opt_cfg.get("min_free_gb", MIN_FREE_GB))

    t0 = time.perf_counter()
    workdir = out_dir / f"work_{preset}_{phase}"
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    env = sqm_thread_env(threads)
    status = "ok"
    val = np.nan
    try:
        shell_caps = opt_cfg.get("caps", "single")
        shell_pdb = out_dir / "capped.pdb"
        # All presets/phases for one frame share capped.pdb. Serialize its
        # initial build so workers cannot read a partially written PDB/sidecar.
        with open(out_dir / ".capped.lock", "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            _build_capped(snapshot, shell_pdb, n_chain_a, protein_res_count, cutoff,
                          max_waters=opt_cfg.get("max_waters"),
                          polar_waters=opt_cfg.get("polar_waters", False),
                          interface_mode=opt_cfg.get("interface_mode", "distance"),
                          dasa_resids=opt_cfg.get("dasa_resids"),
                          force=force, caps=shell_caps,
                          water_radius=opt_cfg.get("water_radius"),
                          water_select=opt_cfg.get("water_select"))
        meta = TI.json.loads(shell_pdb.with_suffix(".trunc.json").read_text())
        formal_qa, formal_qb = meta["charge_a"], meta["charge_b"]
        charge_a, charge_b = formal_qa, formal_qb
        total_charge = charge_a + charge_b
        sum_z = meta.get("sum_z")

        geom = workdir / "geom.pdb"
        tleap_charges = None

        if phase == "mm_sp":
            # The MM-minimized geometry depends only on the frame and MM
            # settings, so it is cached once per frame and shared by all presets.
            mm_tag = f"igb{opt_cfg.get('mm_igb', 7)}_{opt_cfg.get('mm_radii', 'bondi')}"
            cache_geom = out_dir / f".mm_{mm_tag}.pdb"
            cache_meta = out_dir / f".mm_{mm_tag}.json"
            with open(out_dir / f".mm_{mm_tag}.lock", "w") as mm_lock:
                fcntl.flock(mm_lock, fcntl.LOCK_EX)
                if not force and cache_geom.exists() and cache_meta.exists():
                    cached_mm = json.loads(cache_meta.read_text())
                    shutil.copy(cache_geom, geom)
                    tleap_qa = cached_mm["charge_a"]
                    tleap_qb = cached_mm["charge_b"]
                    sum_z_mm = cached_mm.get("sum_z")
                    geom_out, mm_status = geom, "ok"
                else:
                    geom_out, tleap_qa, tleap_qb, sum_z_mm, mm_status = MMGB.run_mm_gb_min(
                        shell_pdb, workdir, MOPAC_DIR,
                        maxcyc=opt_cfg.get("mm_maxcyc", 1000),
                        ncyc=opt_cfg.get("mm_ncyc", 300),
                        igb=opt_cfg.get("mm_igb", 7),
                        radii=opt_cfg.get("mm_radii", "bondi"))
                    if mm_status == "ok":
                        tmp_geom = cache_geom.with_suffix(".pdb.tmp")
                        tmp_meta = cache_meta.with_suffix(".json.tmp")
                        shutil.copy(geom_out, tmp_geom)
                        tmp_meta.write_text(json.dumps({
                            "charge_a": tleap_qa,
                            "charge_b": tleap_qb,
                            "sum_z": sum_z_mm,
                        }))
                        os.replace(tmp_geom, cache_geom)
                        os.replace(tmp_meta, cache_meta)
            if mm_throttle is not None:
                mm_throttle.release()
                mm_throttle = None
            if mm_status != "ok":
                log_path.write_text(
                    f"EXIT=1\nphase={phase}\nstatus={mm_status}\nsecs="
                    f"{round(time.perf_counter() - t0, 1)}\n")
                return key, np.nan, round(time.perf_counter() - t0, 1), mm_status
            if sum_z_mm is not None:
                sum_z = sum_z_mm
            status = "ok"
            tleap_charges = (tleap_qa, tleap_qb)
        elif phase == "md_sp":
            # Short restrained GB MD on the same tleap fragment as mm_sp, then an
            # SQM single point on the last frame. Cached per (igb, radii, nstlim).
            md_tag = (f"igb{opt_cfg.get('mm_igb', 8)}_{opt_cfg.get('mm_radii', 'mbondi3')}"
                      f"_n{opt_cfg.get('md_nstlim', 5000)}")
            cache_geom = out_dir / f".md_{md_tag}.pdb"
            cache_meta = out_dir / f".md_{md_tag}.json"
            with open(out_dir / f".md_{md_tag}.lock", "w") as md_lock:
                fcntl.flock(md_lock, fcntl.LOCK_EX)
                if not force and cache_geom.exists() and cache_meta.exists():
                    cached_md = json.loads(cache_meta.read_text())
                    shutil.copy(cache_geom, geom)
                    tleap_qa = cached_md["charge_a"]
                    tleap_qb = cached_md["charge_b"]
                    sum_z_md = cached_md.get("sum_z")
                    geom_out, md_status = geom, "ok"
                else:
                    geom_out, tleap_qa, tleap_qb, sum_z_md, md_status = MMGB.run_mm_gb_md(
                        shell_pdb, workdir, MOPAC_DIR,
                        nstlim=opt_cfg.get("md_nstlim", 5000),
                        dt=opt_cfg.get("md_dt", 0.002),
                        igb=opt_cfg.get("mm_igb", 8),
                        radii=opt_cfg.get("mm_radii", "mbondi3"))
                    if md_status == "ok":
                        tmp_geom = cache_geom.with_suffix(".pdb.tmp")
                        tmp_meta = cache_meta.with_suffix(".json.tmp")
                        shutil.copy(geom_out, tmp_geom)
                        tmp_meta.write_text(json.dumps({
                            "charge_a": tleap_qa,
                            "charge_b": tleap_qb,
                            "sum_z": sum_z_md,
                        }))
                        os.replace(tmp_geom, cache_geom)
                        os.replace(tmp_meta, cache_meta)
            if mm_throttle is not None:
                mm_throttle.release()
                mm_throttle = None
            if md_status != "ok":
                log_path.write_text(
                    f"EXIT=1\nphase={phase}\nstatus={md_status}\nsecs="
                    f"{round(time.perf_counter() - t0, 1)}\n")
                return key, np.nan, round(time.perf_counter() - t0, 1), md_status
            if sum_z_md is not None:
                sum_z = sum_z_md
            status = "ok"
            tleap_charges = (tleap_qa, tleap_qb)
        elif not meta.get("electrons_even", True):
            log_path.write_text(
                f"EXIT=0\nphase={phase}\nstatus=odd_electron_skip\nsecs=0.0\n"
                f"charge_a={charge_a} charge_b={charge_b} sum_z={sum_z}\n"
                "STDOUT:\n(skipped: odd electron parity, not a closed-shell singlet)\n")
            return key, np.nan, round(time.perf_counter() - t0, 1), "odd_electron_skip"
        elif phase == "opt":
            shutil.copy(shell_pdb, geom)
            mobile = [] if opt_cfg.get("opt_unconstrained") else meta.get("mobile_atoms", [])
            opt_yaml = workdir / "opt.yaml"
            relaxed = workdir / "relaxed.pdb"
            # Adaptive charge search as for the interaction single point. The
            # optimization is one complex job, so only the total charge matters;
            # MOZYME may reject the formal qa+qb ("Mopac thinks the charge is wrong").
            known_q, tried_pairs, tried_tot = {}, set(), set()
            coords_src = None
            opt_err = ""
            while True:
                guess = QM.next_charge_guess(
                    formal_qa, formal_qb, sum_z, known_q, tried_pairs,
                    meta.get("sum_z_a"), meta.get("sum_z_b"))
                if guess is None:
                    break
                qa, qb = guess
                qtot = qa + qb
                tried_pairs.add(guess)
                if qtot in tried_tot:
                    continue
                tried_tot.add(qtot)
                job_mopac = workdir / "job_MOPAC"
                if job_mopac.exists():
                    shutil.rmtree(job_mopac, ignore_errors=True)
                QM.write_optimize_yaml(
                    opt_yaml, geom.name, preset, MOPAC_PATH, threads,
                    mobile, qtot, restart_pdb="optimized.pdb",
                    optimizer=opt_cfg["optimizer"], opt_quality=opt_cfg["opt_quality"],
                    maxcycles=opt_cfg["maxcycles"], cosmo_nspa=opt_cfg.get("cosmo_nspa"),
                )
                out_o, err_o, code_o = safe_run(f"cuby4 {opt_yaml.name}", workdir, env)
                opt_err = err_o
                for name in ("optimized.pdb", "optimized.xyz", "last.xyz"):
                    cand = workdir / name
                    if cand.exists() and cand.stat().st_size > 0:
                        coords_src = cand
                        break
                if coords_src is not None:
                    break
                if not _sqm_retryable(err_o):
                    break
                known_q.update(QM.mozyme_computed_charges(workdir))
            if coords_src is None:
                status = "opt_failed_sp_fallback"
                shutil.copy(shell_pdb, geom)
            else:
                try:
                    QM.overlay_xyz_onto_pdb(shell_pdb, coords_src, relaxed)
                    shutil.copy(relaxed, geom)
                except Exception as exc:  # noqa: BLE001
                    status = f"opt_overlay_fail:{exc}"
                    shutil.copy(shell_pdb, geom)
        else:
            shutil.copy(shell_pdb, geom)

        # Single-point interaction on the (possibly relaxed) geometry.
        yaml_f = workdir / "int.yaml"
        mm_keep = ("geom.pdb", "frag.pdb", "frag.prmtop", "frag.inpcrd",
                   "frag_gb.prmtop", "min.rst7", "min.pdb", "leap.log",
                   "min.out", "min.in", "tleap.in", "radii.in", "topdb.in", "nc.in",
                   "md.rst7", "md.pdb", "md.out", "md.in", "heat.rst7", "heat.in",
                   "heat.out")
        extra_seeds = ()
        if phase in ("mm_sp", "md_sp"):
            # Formal truncate charges first; tleap netCharge is a fallback only.
            seeds = [(formal_qa, formal_qb)]
            tq = tleap_charges
            if tq and tq != (formal_qa, formal_qb):
                seeds.append(tq)
            extra_seeds = tuple(seeds)

        water_assign = opt_cfg.get("water_assign", "receptor")
        sum_z_a, sum_z_b = meta.get("sum_z_a"), meta.get("sum_z_b")
        if sum_z_a is None or sum_z_b is None:
            sum_z_a, sum_z_b = QM.fragment_nuclear_charges(shell_pdb, water_assign)

        def _sqm_on(geom_path, geom_label):
            nonlocal out_c, err_c, code_c, used_qa, used_qb, status
            # Adaptive charge search: MOZYME may reject the submitted charge in
            # favour of its own Lewis-structure charge, and each fragment runs
            # as an independent closed-shell singlet, so parity must hold per
            # fragment. A rejected attempt reports the charge MOZYME computed,
            # which constrains the next guess.
            known, tried = {}, set()
            attempt = 0
            while True:
                guess = QM.next_charge_guess(charge_a, charge_b, sum_z, known,
                                             tried, sum_z_a, sum_z_b,
                                             extra_seeds=extra_seeds)
                if guess is None:
                    break
                qa, qb = guess
                tried.add(guess)
                if attempt:
                    _clean_workdir(workdir, keep=mm_keep if phase == "mm_sp" else ("geom.pdb",))
                attempt += 1
                QM.write_interaction_yaml(yaml_f, geom_path, preset, MOPAC_PATH, threads,
                                          qa, qb, cosmo_nspa=opt_cfg.get("cosmo_nspa"),
                                          water_assign=water_assign)
                out_c, err_c, code_c = safe_run(f"cuby4 {yaml_f.name}", workdir, env)
                used_qa, used_qb = qa, qb
                if code_c == 0:
                    return True
                if not _sqm_retryable(err_c):
                    return False
                known.update(QM.mozyme_computed_charges(workdir))
            if phase in ("mm_sp", "md_sp") and geom_path.resolve() != (workdir / "geom.pdb").resolve():
                status = "mm_geom_sqm_fail"
            return False

        out_c = err_c = ""
        code_c = 1
        used_qa, used_qb = charge_a, charge_b
        if not _sqm_on(geom, "mm_geom"):
            if phase in ("mm_sp", "md_sp"):
                _clean_workdir(workdir, keep=mm_keep)
                fallback_pdb = workdir / "capped.pdb"
                if not fallback_pdb.exists():
                    shutil.copy(shell_pdb, fallback_pdb)
                if _sqm_on(fallback_pdb, "capped_fallback"):
                    status = f"{phase}_capped_fallback"
        elapsed = round(time.perf_counter() - t0, 1)
        charge_note = (f"charge_a={used_qa} charge_b={used_qb} "
                       f"(formal {formal_qa}/{formal_qb})\n")
        log_path.write_text(f"EXIT={code_c}\nphase={phase}\nstatus={status}\nsecs={elapsed}\n"
                            f"{charge_note}"
                            f"STDOUT:\n{out_c}\nSTDERR:\n{err_c}\n")
        if code_c == 0:
            val = parse_interaction_energy(out_c)
            if status in ("ok", "mm_geom_sqm_fail"):
                status = ("ok" if (used_qa, used_qb) == (formal_qa, formal_qb)
                          else "charge_retry_ok")
        else:
            status = f"cuby_exit_{code_c}" if status in ("ok", "mm_geom_sqm_fail") else status
    except Exception as exc:  # noqa: BLE001
        status = f"crash:{exc}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if throttle is not None:
            throttle.release()
        if mm_throttle is not None:
            mm_throttle.release()

    return key, val, round(time.perf_counter() - t0, 1), status


def _init_worker():
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass


def col_name(preset, phase, hydration):
    return f"{preset}_{phase}_{hydration}"


def write_detailed_csv(out_csv, rows, presets, phases, hydrations):
    fields = ["System", "cutoff", "Status", "n_frames"]
    for preset in presets:
        for phase in phases:
            for hy in hydrations:
                base = col_name(preset, phase, hy)
                fields += [base, f"{base}_std", f"{base}_n"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    global OUT_BASE, RUN_BASE, FF
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", help="PDB IDs (default: all 20). 'validation' = "
                    + ",".join(VALIDATION_SYSTEMS))
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[8.0],
                    help="Heavy-atom interface cutoffs in A (default 8). Sweep e.g. 6 8 10.")
    ap.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS,
                    help="SQM presets. 'classical'/'gas'/'ml'/'default' shortcuts available.")
    ap.add_argument("--phases", nargs="*", default=["sp"],
                    choices=["sp", "mm_sp", "md_sp", "opt"],
                    help="sp = bare SP; mm_sp = MM/GB min then SP; "
                         "md_sp = short restrained GB MD then SP; "
                         "opt = SQM relax then SP.")
    ap.add_argument("--hydration", nargs="*", default=["dry", "wet"], choices=["dry", "wet"])
    ap.add_argument("--frames", type=int, nargs="*", help="Snapshot indices (default 1..10).")
    ap.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS)
    ap.add_argument("--workers", type=int, default=MAX_CPU_WORKERS,
                    help=f"Classical SQM pool size (default {MAX_CPU_WORKERS}).")
    ap.add_argument("--ml-workers", type=int, default=DEFAULT_ML_WORKERS,
                    help=f"Max concurrent PM6-ML jobs (default {DEFAULT_ML_WORKERS}; hard cap).")
    ap.add_argument("--mm-workers", type=int, default=4,
                    help="Max concurrent MM/GB preparation jobs (default 4).")
    ap.add_argument("--threads-per-sqm", type=int, default=THREADS_PER_SQM)
    ap.add_argument("--opt-maxcycles", type=int, default=OPT_MAXCYCLES)
    ap.add_argument("--opt-quality", type=float, default=OPT_QUALITY)
    ap.add_argument("--opt-unconstrained", action="store_true",
                    help="SQM opt with no optimize_region (caps and backbone move).")
    ap.add_argument("--opt-optimizer", default=OPT_OPTIMIZER)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                    help=f"Pause a job's heavy compute until this much RAM is free "
                         f"(default {MIN_FREE_GB}).")
    ap.add_argument("--max-waters", type=int, default=None,
                    help="Keep only the N closest bridging interfacial waters "
                         "(Nwat scan). Default: keep all within cutoff.")
    ap.add_argument("--polar-waters", action="store_true",
                    help="Anchor interfacial waters on POLAR interface residues only "
                         "(POLAR setup of the Nwat-MM/GBSA protocol).")
    ap.add_argument("--interface-mode", choices=["distance", "dasa", "dasa_union"],
                    default="distance",
                    help="Interface residue definition: heavy-atom cutoff, static "
                         "crystal dASA, or per-frame dASA union + water-then-grow "
                         "(dasa_union).")
    ap.add_argument("--grow-radius", type=float, default=4.0,
                    help="dasa_union: include residues within this many A of the "
                         "closest interfacial waters (default 4.0).")
    ap.add_argument("--water-radius", type=float, default=None,
                    help="Water-selection distance for cutoff mode, independent of "
                         "the protein cutoff (defaults to the cutoff).")
    ap.add_argument("--dasa-cutoff", type=float, default=0.5,
                    help="Buried-SASA threshold (A^2) for interface_mode=dasa.")
    ap.add_argument("--dasa-polar", action="store_true",
                    help="Use POLAR-only dASA interface residues (ARG/ASP/GLU/...); "
                         "pairs with --polar-waters for the POLAR setup.")
    ap.add_argument("--mm-maxcyc", type=int, default=1000,
                    help="sander MM/GB minimization maxcyc (mm_sp phase).")
    ap.add_argument("--mm-ncyc", type=int, default=300,
                    help="sander MM/GB minimization ncyc (mm_sp phase).")
    ap.add_argument("--mm-igb", type=int, default=7, choices=[7, 8],
                    help="GB model for sander MM/GB min (mm_sp phase).")
    ap.add_argument("--mm-radii", default="bondi", choices=["bondi", "mbondi3"],
                    help="Intrinsic radii set for MM/GB (mm_sp / md_sp phases).")
    ap.add_argument("--caps", choices=["single", "ace_nme"], default="single",
                    help="Truncation caps: 'single' = BNC/BCC/BCG/BCB (bare SP); "
                         "'ace_nme' = parity-safe ACE/NME (required for mm_sp/md_sp "
                         "on all 20 systems).")
    ap.add_argument("--md-nstlim", type=int, default=5000,
                    help="md_sp Langevin steps after a brief clash-relieving min "
                         "(default 5000 * 0.002 ps = 10 ps).")
    ap.add_argument("--md-dt", type=float, default=0.002,
                    help="md_sp timestep in ps (default 0.002).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute all jobs (ignore cached EXIT=0 logs; rebuild shells).")
    ap.add_argument("--water-assign", choices=["receptor", "partition"],
                    default="receptor",
                    help="'receptor' puts every explicit water in molecule_a (the "
                         "Nwat-MMGBSA convention). 'partition' gives each water to "
                         "the chain it is closest to, so its dominant interaction "
                         "cancels out of the interaction energy.")
    ap.add_argument("--cosmo-nspa", type=int, default=None,
                    help="Add 'mopac_cosmo_nspa: N' (COSMO segments per atom) to the "
                         "YAML for solvated presets; finer cavity = less COSMO noise.")
    ap.add_argument("--out-base", default=None,
                    help="Separate output root (default screening_ff_md_sqm/_qmmm_trunc). "
                         "Use a fresh folder for a variant so jobs are not treated as "
                         "cached against the main run.")
    ap.add_argument("--in-base", default=None,
                    help="Root holding the <SYS>/ff14sb/rep1/{dry,wet}_f*.pdb snapshots "
                         "(default screening_ff_md_sqm). Point at e.g. "
                         "another snapshot tree.")
    ap.add_argument("--rep", default=REP,
                    help="Replica subfolder under <SYS>/<ff>/ (default rep1).")
    ap.add_argument("--ff", default=FF,
                    help="Force-field subfolder under <SYS>/ (default ff14sb).")
    ap.add_argument("--out-csv", default=None, help="Detailed CSV path (default under _qmmm_trunc).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    presets = []
    for p in args.presets:
        if p == "classical":
            presets += CLASSICAL_PRESETS
        elif p == "gas":
            presets += GAS_PRESETS
        elif p == "ml":
            presets += ML_PRESETS
        elif p == "newcorr":
            presets += NEWCORR_PRESETS
        elif p == "default":
            presets += DEFAULT_PRESETS
        elif p == "all":
            presets += ALL_PRESETS
        else:
            presets.append(p)
    seen = set()
    presets = [p for p in presets if not (p in seen or seen.add(p))]
    bad = [p for p in presets if p not in SQM_PRESETS]
    if bad:
        print(f"ERROR: unknown preset(s): {bad}")
        sys.exit(1)
    pm7_warn = {"pm7_cosmo2", "pm7_d3h4x_cosmo2"} & set(presets)
    if pm7_warn:
        print(f"NOTE: {pm7_warn} may overflow MOPAC's energy field on some systems; "
              "failures will be logged as NaN.")

    systems = args.systems or PDB_IDS
    if systems == ["validation"]:
        systems = VALIDATION_SYSTEMS
    bad_sys = [s for s in systems if s not in PDB_IDS]
    if bad_sys:
        print(f"ERROR: unknown system(s): {bad_sys}")
        sys.exit(1)

    frames = args.frames or list(range(1, args.n_snapshots + 1))
    if (args.workers < 1 or args.mm_workers < 1 or args.ml_workers < 1
            or not (1 <= args.ml_workers <= 32)):
        print("ERROR: --workers/--mm-workers >=1 and 1 <= --ml-workers <= 32 required")
        sys.exit(1)

    if args.out_base:
        OUT_BASE = Path(args.out_base).expanduser().resolve()
    if args.in_base:
        RUN_BASE = Path(args.in_base).expanduser().resolve()
    FF = args.ff

    # Per-system A/B split: scale input-PDB chain boundary to each MD snapshot.
    split = {}
    for s in systems:
        inp = PDB_INPUT_DIR / f"{s}_input.pdb"
        snap_dry = RUN_BASE / s / FF / args.rep / "dry_f1.pdb"
        if snap_dry.exists():
            split[s] = resolve_snapshot_chain_split(snap_dry, inp)
        else:
            split[s] = (get_first_chain_nres(inp), get_protein_residue_count(inp))

    # Per-system dASA / dasa_union masks (cached).
    dasa_masks = {}
    if args.interface_mode == "dasa":
        for s in systems:
            dasa_masks[s] = ID.get_mask(s, dasa_cutoff=args.dasa_cutoff,
                                        polar=args.dasa_polar)
    elif args.interface_mode == "dasa_union":
        import dasa_union as DU  # noqa: E402
        nwat = args.max_waters if args.max_waters is not None else 20
        for s in systems:
            nA, protRes = split[s]
            dasa_masks[s] = DU.get_locked_resids(
                s, args.rep, RUN_BASE, nA, protRes, frames,
                ff=FF, dasa_cutoff=args.dasa_cutoff, nwat=nwat,
                grow_radius=args.grow_radius,
                polar_waters=args.polar_waters,
            )

    water_select = "closest" if args.interface_mode == "dasa_union" else "cutoff"
    opt_cfg = {"optimizer": args.opt_optimizer, "opt_quality": args.opt_quality,
               "maxcycles": args.opt_maxcycles, "min_free_gb": args.min_free_gb,
               "cosmo_nspa": args.cosmo_nspa, "max_waters": args.max_waters,
               "polar_waters": args.polar_waters, "force": args.force,
               "interface_mode": args.interface_mode, "rep": args.rep,
               "mm_maxcyc": args.mm_maxcyc, "mm_ncyc": args.mm_ncyc,
               "mm_igb": args.mm_igb, "mm_radii": args.mm_radii,
               "caps": args.caps, "md_nstlim": args.md_nstlim,
               "md_dt": args.md_dt,
               "water_assign": args.water_assign,
               "water_radius": args.water_radius,
               "water_select": water_select,
               "grow_radius": args.grow_radius,
               "opt_unconstrained": args.opt_unconstrained}

    has_ml = any(is_ml_preset(p) for p in presets)
    print(f"Systems: {len(systems)} | cutoffs: {args.cutoffs} | presets: {len(presets)} "
          f"({presets[:3]}{'...' if len(presets) > 3 else ''})")
    print(f"phases: {args.phases} | hydration: {args.hydration} | frames: {len(frames)}")
    print(f"interface_mode: {args.interface_mode} | dasa_cutoff: {args.dasa_cutoff} | "
          f"dasa_polar: {args.dasa_polar} | polar_waters: {args.polar_waters} | "
          f"caps: {args.caps} | force: {args.force}")
    print(f"workers: {args.workers} | PM6-ML cap: {args.ml_workers if has_ml else 'n/a'} "
          f"| MM/GB cap: {args.mm_workers} | threads/job: {args.threads_per_sqm}")

    tasks = []
    for system in systems:
        nA, protRes = split[system]
        for cutoff in args.cutoffs:
            for frame in frames:
                for hy in args.hydration:
                    for preset in presets:
                        for phase in args.phases:
                            tasks.append((system, cutoff, frame, hy, preset, phase,
                                          nA, protRes, args.threads_per_sqm, None,
                                          {**opt_cfg, "dasa_resids": dasa_masks.get(system)}))
    print(f"Total SQM jobs: {len(tasks)}")
    if args.dry_run:
        for t in tasks[:8]:
            print("  ", t[:6])
        if len(tasks) > 8:
            print(f"  ... and {len(tasks) - 8} more")
        return

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    # PM6-ML jobs use a separate executor: a semaphore inside one shared pool
    # would let blocked ML tasks occupy workers and starve classical jobs.
    manager = Manager()
    mm_sem = manager.Semaphore(args.mm_workers)
    tasks = [t[:9] + (None, {**t[10], "mm_sem": mm_sem}) for t in tasks]
    classical_tasks = [t for t in tasks if not is_ml_preset(t[4])]
    ml_tasks = [t for t in tasks if is_ml_preset(t[4])]

    # Per-frame values keyed by (system, cutoff, hydration, preset, phase).
    vals = {}
    secs = {}
    statuses = {}
    t_start = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as classical_ex, \
            ProcessPoolExecutor(max_workers=args.ml_workers, initializer=_init_worker) as ml_ex:
        futs = [classical_ex.submit(run_one, t) for t in classical_tasks]
        futs += [ml_ex.submit(run_one, t) for t in ml_tasks]
        for fut in as_completed(futs):
            key, val, sec, status = fut.result()
            (system, cutoff, hy, preset, phase, frame) = key
            k = (system, cutoff, hy, preset, phase)
            if not (isinstance(val, float) and np.isnan(val)):
                vals.setdefault(k, []).append(val)
            secs[k] = secs.get(k, 0.0) + sec
            statuses.setdefault((system, cutoff), []).append(status)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                el = (time.perf_counter() - t_start) / 60
                print(f"  [{done}/{len(tasks)}] {el:.1f} min | last: {system} c{cutoff:g} "
                      f"{hy} {preset} {phase} f{frame} -> {val} ({status})")

    # Aggregate per (System, cutoff).
    rows_by_cutoff = {}
    for system in systems:
        for cutoff in args.cutoffs:
            row = {"System": system, "cutoff": cutoff, "Status": "ok"}
            st = statuses.get((system, cutoff), [])
            row["Status"] = "ok" if any(s in ("ok", "cached") for s in st) else (st[0] if st else "none")
            nframes = 0
            for preset in presets:
                for phase in args.phases:
                    for hy in args.hydration:
                        v = vals.get((system, cutoff, hy, preset, phase), [])
                        base = col_name(preset, phase, hy)
                        row[base] = float(np.mean(v)) if v else np.nan
                        row[f"{base}_std"] = float(np.std(v)) if v else np.nan
                        row[f"{base}_n"] = len(v)
                        nframes = max(nframes, len(v))
            row["n_frames"] = nframes
            rows_by_cutoff.setdefault(cutoff, []).append(row)

    # One CSV per cutoff; columns <preset>_<phase>_<hydration> with _std and _n.
    for cutoff, rows in rows_by_cutoff.items():
        if args.out_csv and len(args.cutoffs) == 1:
            out_csv = Path(args.out_csv)
        else:
            out_csv = OUT_BASE / f"detailed_qmmm_trunc_c{cutoff:g}.csv"
        write_detailed_csv(out_csv, rows, presets, args.phases, args.hydration)
        print(f"Wrote {out_csv}")

    total_h = (time.perf_counter() - t_start) / 3600
    print(f"\nDone: {done} SQM jobs in {total_h:.2f} h")


if __name__ == "__main__":
    main()
