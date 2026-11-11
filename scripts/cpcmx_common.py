#!/usr/bin/env python3
"""GFN2-xTB/CPCM-X interaction energies G(AB) - G(A) - G(B) in kcal/mol.

CPCM-X (Stahn, Ehlert, Grimme, J. Phys. Chem. A 2023, 127, 7036):
G = E_gas + dG_solv, dG_solv = dG_IS + dG_res + dG_CDS + dG_corr + ...
E_gas is a gas-phase single point; dG_solv comes from the standalone cpx program
on an xtb conductor (.cosmo) file, with dG_IS = E(conductor) - E_gas. The second
SCF that ``xtb --cpcmx`` restarts from the conductor density is not used.
Writes one JSON result per frame under --out-base.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import xtb_runner as X
from paths import CPCMX_DB, CPX_EXE, XTB_HOME

XTB = XTB_HOME / "bin" / "xtb"
CPX = Path(shutil.which(CPX_EXE) or CPX_EXE)
CPX_DB = CPCMX_DB
HARTREE_KCAL = X.HARTREE_KCAL
# HOMO-LUMO gap threshold. Collapsed GFN2 gaps on these fragments are ~0.1 eV,
# converged ones > 2 eV.
GAP_MIN_EV = 0.5

_CPX_EH = {
    "dG_is": re.compile(r"Ideal State \(dG_is\):\s+(\S+)"),
    "dG_cc": re.compile(r"Averaging correction \(dG_cc\):\s+(\S+)"),
    "dG_res": re.compile(r"restoring free energy \(dG_res\):\s+(\S+)"),
    "dG_CDS": re.compile(r"SMD Contribution \(dG_CDS\):\s+(\S+)"),
    "dG_corr": re.compile(r"Standard state correction:\s+(\S+)"),
    "dG_shift": re.compile(r"Systematic empirical shift \(dG_shift\)\s+(\S+)"),
    "Gsolv": re.compile(r"solvation free energy:\s+(\S+)"),
}
def _rss_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


def env_for(threads, stack):
    env = X.xtb_env(threads)
    env["OMP_STACKSIZE"] = stack
    env["MALLOC_ARENA_MAX"] = "2"
    env["CPXHOME"] = str(CPX_DB.parent.parent) + "/"
    return env


def _run(cmd, cwd, env, timeout):
    import subprocess
    import threading
    try:
        p = subprocess.Popen(
            cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, preexec_fn=X._unlimited_stack)
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}:{exc}", 0, f"{type(exc).__name__}"
    peak = [0]
    stop = threading.Event()

    def _sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _rss_kb(p.pid))
            stop.wait(0.5)

    threading.Thread(target=_sample, daemon=True).start()
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        stop.set()
        return out or "", "timeout", peak[0], "timeout"
    stop.set()
    text = out or ""
    if p.returncode != 0 and "TOTAL ENERGY" not in text and "solvation free energy" not in text:
        tail = "\n".join(text.strip().splitlines()[-6:])
        return text, tail[:400], peak[0], f"exit_{p.returncode}"
    return text, "", peak[0], ""


def parse_sp(text):
    m = None
    for m in X._ENERGY_RE.finditer(text):
        pass
    if m is None:
        return None
    gaps = list(X._GAP_RE.finditer(text))
    rec = {"E_Eh": float(m.group(1))}
    if gaps:
        rec["gap_eV"] = float(gaps[-1].group(1))
    return rec


def parse_cpx(text):
    parts = {}
    for key, pat in _CPX_EH.items():
        ms = list(pat.finditer(text))
        if ms:
            parts[key] = float(ms[-1].group(1))
    return parts


def write_cpx_input(path, solute):
    # cpx reads each input line into a 100-character buffer, so paths must be short.
    path.write_text(
        "\n".join([
            str(CPX_DB / "crs.param_h2o"),
            str(CPX_DB / "smd_h2o"),
            "crs",
            "# conductor cosmo, gas energy in gas.energy",
            str(CPX_DB / "water.cosmo"),
            solute,
            "water 0.4",
            "298.15",
            "",
        ])
    )


def one_species(xyz, charge, work, env, timeout):
    """Gas-phase SP, conductor SP and cpx for one species. Returns (record, err, maxrss_kb)."""
    work = Path(work)
    gas_dir = work / "gas"
    cos_dir = work / "cos"
    gas_dir.mkdir(parents=True, exist_ok=True)
    cos_dir.mkdir(parents=True, exist_ok=True)
    xyz = Path(xyz)
    shutil.copy2(xyz, gas_dir / xyz.name)
    shutil.copy2(xyz, cos_dir / xyz.name)
    rss = 0

    text, err, r, fail = _run(
        [str(XTB), xyz.name, "--sp", "--gfn", "2", "--chrg", str(int(charge)),
         "--acc", "1.0", "--norestart"],
        gas_dir, env, timeout)
    rss = max(rss, r)
    (gas_dir / "xtb.out").write_text(text)
    if fail and parse_sp(text) is None:
        return None, f"gas:{err or fail}", rss
    gas = parse_sp(text)
    if gas is None:
        return None, "gas:no_energy", rss

    text, err, r, fail = _run(
        [str(XTB), xyz.name, "--sp", "--gfn", "2", "--chrg", str(int(charge)),
         "--acc", "1.0", "--norestart", "--cosmo", "infinity"],
        cos_dir, env, timeout)
    rss = max(rss, r)
    (cos_dir / "xtb.out").write_text(text)
    cos = parse_sp(text)
    if cos is None:
        return None, f"cosmo:{err or fail or 'no_energy'}", rss
    cosmo = cos_dir / "xtb.cosmo"
    if not cosmo.exists():
        return None, "cosmo:no_cosmo_file", rss

    (cos_dir / "gas.energy").write_text(f"{gas['E_Eh']:.16f}\n")
    write_cpx_input(cos_dir / "csx.input", "xtb.cosmo")
    text, err, r, fail = _run([str(CPX), "--inp", "csx.input"], cos_dir, env, timeout)
    rss = max(rss, r)
    (cos_dir / "cpx.out").write_text(text)
    parts = parse_cpx(text)
    if "Gsolv" not in parts:
        return None, f"cpx:{err or fail or 'no_gsolv'}", rss
    rec = {
        "Egas_Eh": gas["E_Eh"],
        "gap_gas_eV": gas.get("gap_eV"),
        "Ecosmo_Eh": cos["E_Eh"],
        "gap_cosmo_eV": cos.get("gap_eV"),
        **parts,
    }
    return rec, "", rss


def score_frame(capped_pdb, work, charge_a, charge_b, threads, stack, timeout):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = env_for(threads, stack)
    atoms_a, atoms_b = X.split_chains(capped_pdb)
    if not atoms_a or not atoms_b:
        return None, "missing_chain", 0
    X.write_xyz(atoms_a + atoms_b, work / "ab.xyz", "complex")
    X.write_xyz(atoms_a, work / "a.xyz", "partner A")
    X.write_xyz(atoms_b, work / "b.xyz", "partner B")

    def species(name, xyz, q0):
        rss = 0
        last = ""
        for dq in (0, -2, 2, -4, 4):
            rec, err, r = one_species(work / xyz, q0 + dq, work / name, env, timeout)
            rss = max(rss, r)
            if rec is not None:
                return rec, q0 + dq, "", rss
            last = err
            if err and "timeout" in err:
                break
        return None, q0, last, rss

    rec_a, qa, err, rss = species("a", "a.xyz", charge_a)
    if rec_a is None:
        return None, f"A:{err}", rss
    rec_b, qb, err, r = species("b", "b.xyz", charge_b)
    rss = max(rss, r)
    if rec_b is None:
        return None, f"B:{err}", rss
    rec_ab, qab, err, r = species("ab", "ab.xyz", qa + qb)
    rss = max(rss, r)
    if rec_ab is None:
        return None, f"AB:{err}", rss
    rec = {"a": rec_a, "b": rec_b, "ab": rec_ab}
    eg = (rec["ab"]["Egas_Eh"] - rec["a"]["Egas_Eh"] - rec["b"]["Egas_Eh"]) * HARTREE_KCAL
    gs = (rec["ab"]["Gsolv"] - rec["a"]["Gsolv"] - rec["b"]["Gsolv"]) * HARTREE_KCAL
    row = {
        "energy": eg + gs,
        "energy_gas": eg,
        "Gsolv_cpcmx": gs,
        "charge_a_used": qa,
        "charge_b_used": qb,
        "maxrss_kb": rss,
    }
    for key in ("dG_is", "dG_cc", "dG_res", "dG_CDS", "dG_corr", "dG_shift"):
        if key in rec["ab"] and key in rec["a"] and key in rec["b"]:
            row[key] = (rec["ab"][key] - rec["a"][key] - rec["b"][key]) * HARTREE_KCAL
    for name in ("a", "b", "ab"):
        row[f"gap_gas_{name}_eV"] = rec[name].get("gap_gas_eV")
        row[f"gap_cosmo_{name}_eV"] = rec[name].get("gap_cosmo_eV")
    # Gas-phase gaps on these charged fragments can collapse (~0.1 eV), but E_gas
    # cancels: cpx sets dG_IS = E(conductor) - E_gas and G adds E_gas back, so
    # G = E(conductor) + restoring, CDS and correction terms. Only a collapsed
    # conductor gap invalidates G.
    cosmo_gaps = [rec[n].get("gap_cosmo_eV") for n in ("a", "b", "ab")]
    collapsed = any(g is not None and g < GAP_MIN_EV for g in cosmo_gaps)
    status = "gap_collapse" if collapsed else (
        "ok" if (qa, qb) == (charge_a, charge_b) else "charge_retry_ok")
    return row, status, rss


def run_one(task):
    system, frame, shell_dir, out_dir, threads, stack, timeout = task
    res_json = Path(out_dir) / system / f"f{frame}" / "gfn2_cpcmx_corrected.json"
    if res_json.exists():
        try:
            d = json.loads(res_json.read_text())
            if d.get("status") in ("ok", "charge_retry_ok", "gap_collapse"):
                return system, frame, d.get("energy"), d.get("status"), d.get("maxrss_kb") or 0
        except Exception:  # noqa: BLE001
            pass
    shell = Path(shell_dir)
    capped, meta_f = shell / "capped.pdb", shell / "capped.trunc.json"
    if not capped.exists() or not meta_f.exists():
        return system, frame, None, "missing_shell", 0
    meta = json.loads(meta_f.read_text())
    work = Path(f"/tmp/cpxw_{os.getpid()}_{system}_f{frame}")
    t0 = time.time()
    try:
        row, status, rss = score_frame(
            capped, work, meta["charge_a"], meta["charge_b"], threads, stack, timeout)
    except Exception as exc:  # noqa: BLE001
        row, status, rss = None, f"exc:{type(exc).__name__}:{exc}"[:200], 0
    dt = time.time() - t0
    res_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system, "frame": frame, "variant": "gfn2_cpcmx_corrected",
        "energy": None if row is None else row["energy"],
        "status": status, "seconds": round(dt, 1),
        "n_atoms": meta.get("n_atoms"),
        "charge_a": meta["charge_a"], "charge_b": meta["charge_b"],
        "maxrss_kb": rss,
    }
    if row:
        payload.update(row)
    tmp = res_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(res_json)
    shutil.rmtree(work, ignore_errors=True)
    return system, frame, payload["energy"], status, rss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shell-base", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--cutoff", default="c8")
    ap.add_argument("--hydration", default="wet")
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stack", default="512M",
                    help="OMP_STACKSIZE per thread. 4G reserves too much "
                         "virtual memory when many jobs run at once.")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    shell_base = Path(args.shell_base)
    out_base = Path(args.out_base)
    if args.systems:
        systems = args.systems
    else:
        systems = sorted(p.name for p in shell_base.iterdir() if p.is_dir() and not p.name.startswith("_"))

    def frames_for(system):
        d = shell_base / system / args.cutoff / args.hydration
        found = []
        if d.is_dir():
            for p in d.glob("f*"):
                if (p / "capped.pdb").exists():
                    try:
                        found.append(int(p.name[1:]))
                    except ValueError:
                        pass
        if args.frames:
            found = [f for f in found if f in args.frames]
        return sorted(found)

    tasks = []
    for s in systems:
        for f in frames_for(s):
            sd = shell_base / s / args.cutoff / args.hydration / f"f{f}"
            tasks.append((s, f, str(sd), str(out_base), args.threads, args.stack, args.timeout))

    cores = args.workers * args.threads
    print(f"CPCM-X: {len(tasks)} frames, "
          f"{args.workers} workers x {args.threads} threads = {cores} cores, "
          f"OMP_STACKSIZE={args.stack}", flush=True)
    print(f"shells: {shell_base}\nout:    {out_base}", flush=True)

    os.nice(10)
    done = 0
    t0 = time.time()
    peak = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            s, f, e, st, rss = fut.result()
            peak = max(peak, rss or 0)
            done += 1
            if done % 10 == 0 or done == len(tasks) or st not in ("ok", "charge_retry_ok"):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - done) / max(rate, 1e-9) / 60
                mem = ""
                try:
                    avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0])
                    mem = f" avail={avail/1024/1024:.0f}GB"
                except Exception:  # noqa: BLE001
                    pass
                print(f"  [{done}/{len(tasks)}] {s} f{f}: "
                      f"{'NaN' if e is None else f'{e:.1f}'} ({st}) "
                      f"jobRSS={rss/1024/1024:.1f}GB peak={peak/1024/1024:.1f}GB"
                      f"{mem} ETA {eta:.0f} min", flush=True)
    print(f"peak single-job RSS {peak/1024/1024:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
