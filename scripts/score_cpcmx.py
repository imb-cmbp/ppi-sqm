#!/usr/bin/env python3
"""GFN2-xTB/CPCM-X interaction energies G(AB) - G(A) - G(B) in kcal/mol.

Uses the CPCM-X routines in cpcmx_common.py. Partners are scored at their native
charges only; a species that fails there fails the frame. No gas-phase SCF is run:
E_gas cancels in G (dG_IS = E(conductor) - E_gas - thermo), so gas.energy is set to
E(conductor). A conductor SCF with a HOMO-LUMO gap below GAP_QC_EV is repeated and
accepted only when two runs agree within E_TOL_EH. OpenMP/MKL dynamic threading is
disabled, and workers wait for free memory before each SCF.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cpcmx_common as V1
import xtb_runner as X

HARTREE_KCAL = X.HARTREE_KCAL
GAP_QC_EV = 0.5
E_TOL_EH = 1e-4
MAX_RUNS = 3
MIN_FREE_KB = 30 * 1024 * 1024
COMPONENTS = ("dG_is", "dG_cc", "dG_res", "dG_CDS", "dG_corr", "dG_shift")

from paths import WORK

REPO = WORK
SCR = REPO / "screening_ff_md_sqm"
SETS = {
    "iface": SCR / "_qmmm_trunc_propka_clust_iface_sc_gt5",
    "aa": SCR / "_qmmm_trunc_propka_clust_aa_gt5",
    "bb": SCR / "_qmmm_trunc_propka_clust_bb_gt5",
    "contacts": SCR / "_qmmm_trunc_propka_clust_contacts_gt5",
    "even10": SCR / "_qmmm_trunc_propka",
}
REPS = ("r1", "r2", "r3", "r4", "r5")
OUT_ROOT = SCR / "_xtb_cpcmx_v2_propka"
V1_IFACE = SCR / "_xtb_cpcmx_corrected_propka_iface"


def env_for(threads, stack):
    env = V1.env_for(threads, stack)
    env.update({
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "KMP_DETERMINISTIC_REDUCTION": "yes",
    })
    return env


def _mem_free_kb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return MIN_FREE_KB


def _wait_for_memory():
    while _mem_free_kb() < MIN_FREE_KB:
        time.sleep(15)


def _keep(text, path):
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def conductor_sp(xyz, charge, run_dir, env, timeout):
    """One --cosmo infinity single point. Returns (record, err, rss)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xyz, run_dir / xyz.name)
    _wait_for_memory()
    text, err, rss, fail = V1._run(
        [str(V1.XTB), xyz.name, "--sp", "--gfn", "2", "--chrg", str(int(charge)),
         "--acc", "1.0", "--norestart", "--cosmo", "infinity"],
        run_dir, env, timeout)
    (run_dir / "xtb.out").write_text(text)
    rec = V1.parse_sp(text)
    if rec is None or "did not converge" in text or "not converged" in text:
        return None, err or fail or "no_energy_or_unconverged", rss
    if not (run_dir / "xtb.cosmo").exists():
        return None, "no_cosmo_file", rss
    return rec, "", rss


def one_species(xyz, charge, work, env, timeout, keep_dir, name):
    """Conductor SCF with reproducibility QC, then cpx. Returns (rec, err, rss)."""
    work = Path(work)
    runs, rss, last_err = [], 0, ""
    chosen = None
    for k in range(MAX_RUNS):
        rec, err, r = conductor_sp(xyz, charge, work / f"cos{k}", env, timeout)
        rss = max(rss, r)
        if rec is None:
            last_err = err
            if "timeout" in err:
                break
            continue
        rec["dir"] = work / f"cos{k}"
        runs.append(rec)
        if len(runs) == 1 and rec.get("gap_eV", 0.0) >= GAP_QC_EV:
            chosen = rec
            break
        for i in range(len(runs) - 1):
            if abs(runs[i]["E_Eh"] - rec["E_Eh"]) < E_TOL_EH:
                chosen = runs[i]
                break
        if chosen is not None:
            break
    for k, rec in enumerate(runs):
        _keep((rec["dir"] / "xtb.out").read_text(), keep_dir / f"{name}_cos{k}.out.gz")
    if chosen is None:
        if not runs:
            return None, f"cosmo:{last_err}", rss
        return None, "scf_unstable:" + ",".join(f"{r['E_Eh']:.6f}" for r in runs), rss

    cos_dir = chosen["dir"]
    (cos_dir / "gas.energy").write_text(f"{chosen['E_Eh']:.16f}\n")
    V1.write_cpx_input(cos_dir / "csx.input", "xtb.cosmo")
    text, err, r, fail = V1._run([str(V1.CPX), "--inp", "csx.input"], cos_dir, env, timeout)
    rss = max(rss, r)
    _keep(text, keep_dir / f"{name}_cpx.out.gz")
    parts = V1.parse_cpx(text)
    if "Gsolv" not in parts:
        return None, f"cpx:{err or fail or 'no_gsolv'}", rss
    out = {
        "Ecosmo_Eh": chosen["E_Eh"],
        "gap_cosmo_eV": chosen.get("gap_eV"),
        "n_scf_runs": len(runs),
        "scf_energies_Eh": [r["E_Eh"] for r in runs],
        "G_Eh": chosen["E_Eh"] + parts["Gsolv"],
        **parts,
    }
    return out, "", rss


def score_frame(capped_pdb, work, charge_a, charge_b, threads, stack, timeout, keep_dir):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = env_for(threads, stack)
    atoms_a, atoms_b = X.split_chains(capped_pdb)
    if not atoms_a or not atoms_b:
        return None, "missing_chain", 0
    X.write_xyz(atoms_a + atoms_b, work / "ab.xyz", "complex")
    X.write_xyz(atoms_a, work / "a.xyz", "partner A")
    X.write_xyz(atoms_b, work / "b.xyz", "partner B")
    rec, rss = {}, 0
    for name, xyz, q in (("a", "a.xyz", charge_a), ("b", "b.xyz", charge_b),
                         ("ab", "ab.xyz", charge_a + charge_b)):
        r, err, m = one_species(work / xyz, q, work / name, env, timeout, keep_dir, name)
        rss = max(rss, m)
        if r is None:
            return None, f"{name}:{err}"[:300], rss
        rec[name] = r

    def delta(key):
        return (rec["ab"][key] - rec["a"][key] - rec["b"][key]) * HARTREE_KCAL

    row = {
        "energy": delta("G_Eh"),
        "energy_conductor": delta("Ecosmo_Eh"),
        "post_scf": delta("G_Eh") - delta("Ecosmo_Eh"),
        "maxrss_kb": rss,
    }
    for key in COMPONENTS:
        if all(key in rec[n] for n in ("a", "b", "ab")):
            row[key] = delta(key)
    for n in ("a", "b", "ab"):
        row[f"gap_cosmo_{n}_eV"] = rec[n]["gap_cosmo_eV"]
        row[f"n_scf_runs_{n}"] = rec[n]["n_scf_runs"]
        row[f"scf_energies_{n}_Eh"] = rec[n]["scf_energies_Eh"]
    return row, "ok", rss


def run_one(task):
    set_name, rep, system, frame, shell_dir, out_dir, threads, stack, timeout = task
    out_dir = Path(out_dir)
    res_json = out_dir / "gfn2_cpcmx.json"
    if res_json.exists():
        try:
            d = json.loads(res_json.read_text())
            if d.get("status") == "ok":
                return set_name, rep, system, frame, d.get("energy"), "ok", d.get("maxrss_kb") or 0
        except ValueError:
            pass
    shell = Path(shell_dir)
    capped, meta_f = shell / "capped.pdb", shell / "capped.trunc.json"
    if not capped.exists() or not meta_f.exists():
        return set_name, rep, system, frame, None, "missing_shell", 0
    meta = json.loads(meta_f.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_dir = out_dir / "logs"
    keep_dir.mkdir(exist_ok=True)
    work = Path(f"/tmp/cpx2_{os.getpid()}_{set_name}_{rep}_{system}_f{frame}")
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    try:
        row, status, rss = score_frame(capped, work, int(meta["charge_a"]), int(meta["charge_b"]),
                                       threads, stack, timeout, keep_dir)
    except Exception as exc:  # noqa: BLE001
        row, status, rss = None, f"exc:{type(exc).__name__}:{exc}"[:300], 0
    payload = {
        "system": system, "frame": frame, "rep": rep, "set": set_name,
        "variant": "gfn2_cpcmx_v2",
        "energy": None if row is None else row["energy"],
        "status": status if row is not None else f"failed:{status}",
        "seconds": round(time.time() - t0, 1),
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
    return set_name, rep, system, frame, payload["energy"], payload["status"], rss


def frames_in(shell_root, rep):
    d = shell_root / rep
    out = []
    if not d.is_dir():
        return out
    for sysdir in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
        wet = sysdir / "c8" / "wet"
        if not wet.is_dir():
            continue
        for fd in wet.glob("f*"):
            if (fd / "capped.pdb").exists():
                try:
                    out.append((sysdir.name, int(fd.name[1:]), fd))
                except ValueError:
                    pass
    return out


def iface_needs_rerun(rep, system, frame):
    """Return (rerun, record) for an interface frame scored with cpcmx_common.py.

    Frames with status ok at native charges give the same G here and are reused;
    frames with a collapsed gap, a changed charge or no energy are rescored."""
    f = V1_IFACE / rep / system / f"f{frame}" / "gfn2_cpcmx_corrected.json"
    if not f.exists():
        return True, None
    d = json.loads(f.read_text())
    native = (d.get("charge_a_used") == d.get("charge_a")
              and d.get("charge_b_used") == d.get("charge_b"))
    return (d.get("status") != "ok" or not native or d.get("energy") is None), d


def carry_v1(d, out_dir, rep):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": d["system"], "frame": d["frame"], "rep": rep, "set": "iface",
        "variant": "gfn2_cpcmx_v2", "energy": d["energy"], "status": "ok",
        "source": "reused_identical", "n_atoms": d.get("n_atoms"),
        "charge_a": d["charge_a"], "charge_b": d["charge_b"],
        "energy_conductor": d["energy_gas"] + d["dG_is"],
    }
    for key in COMPONENTS:
        payload[key] = d.get(key)
    for n in ("a", "b", "ab"):
        payload[f"gap_cosmo_{n}_eV"] = d.get(f"gap_cosmo_{n}_eV")
    (out_dir / "gfn2_cpcmx.json").write_text(json.dumps(payload, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=["iface", "aa", "bb", "contacts", "even10"])
    ap.add_argument("--only", nargs="*", help="rep/system/frame triples, e.g. r2/1EMV/6")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stack", default="512M")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    only = None
    if args.only:
        only = {(t.split("/")[0], t.split("/")[1], int(t.split("/")[2])) for t in args.only}
    tasks, carried = [], 0
    for set_name in args.sets:
        root = SETS[set_name]
        for rep in REPS:
            for system, frame, fd in frames_in(root, rep):
                if only is not None and (rep, system, frame) not in only:
                    continue
                out_dir = out_root / set_name / rep / system / f"f{frame}"
                if set_name == "iface" and only is None:
                    redo, d = iface_needs_rerun(rep, system, frame)
                    if not redo:
                        if not (out_dir / "gfn2_cpcmx.json").exists():
                            carry_v1(d, out_dir, rep)
                        carried += 1
                        continue
                tasks.append((set_name, rep, system, frame, str(fd), str(out_dir),
                              args.threads, args.stack, args.timeout))
    # Run sets in the order given, largest complexes first within each set.
    order = {name: i for i, name in enumerate(args.sets)}

    def key(t):
        try:
            n = json.loads((Path(t[4]) / "capped.trunc.json").read_text()).get("n_atoms", 0)
        except (OSError, ValueError):
            n = 0
        return order[t[0]], -n
    tasks.sort(key=key)
    print(f"CPCM-X: {len(tasks)} frames to score, {carried} reused from an earlier identical run, "
          f"{args.workers} workers x {args.threads} threads, sets {args.sets}", flush=True)
    print(f"out: {out_root}", flush=True)
    os.nice(10)
    done, t0, peak = 0, time.time(), 0
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            set_name, rep, s, f, e, st, rss = fut.result()
            done += 1
            peak = max(peak, rss or 0)
            counts[st.split(":")[0]] = counts.get(st.split(":")[0], 0) + 1
            if done % 25 == 0 or done == len(tasks) or st != "ok":
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - done) / max(rate, 1e-9) / 60
                print(f"  [{done}/{len(tasks)}] {set_name} {rep} {s} f{f}: "
                      f"{'NaN' if e is None else f'{e:.1f}'} ({st[:120]}) "
                      f"jobRSS={rss/1048576:.1f}GB peak={peak/1048576:.1f}GB "
                      f"avail={_mem_free_kb()/1048576:.0f}GB ETA {eta:.0f} min", flush=True)
    print(f"status counts {counts}", flush=True)
    print("CPCMX_DONE", flush=True)


if __name__ == "__main__":
    main()
