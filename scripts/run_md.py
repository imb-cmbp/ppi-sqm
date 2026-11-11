#!/usr/bin/env python3
"""Run the staged MD protocol of Maffucci & Contini (2016) with pmemd.cuda.

Stages (do_all_MD_HPC_5.4 order): md1-6 (minimization/equilibration), heat1-6
(0 to 300 K), md7-12 (restrained equilibration), then 4 x 1 ns production (md13).
Each system runs on one GPU; systems are distributed over the GPUs in --gpus.
Stages whose .rst file already exists are skipped, so runs can be resumed.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BENCHMARK, WORK  # noqa: E402

REPO = WORK
from amber_setup import amber_env, gpu_engine  # noqa

ART = BENCHMARK
PROTO = ART / "protocol" / "protocol"
SYS = REPO / "md" / "systems"

PDB_IDS = ["1ACB", "1ZHI", "1AVX", "2HLE", "1AY7", "2HRK", "1BVN", "2OOB", "1EMV",
           "2OUL", "1FLE", "2SIC", "1GLA", "2SNI", "1KAC", "2UUY", "1R0R", "3BZD",
           "1YVB", "3SGB"]

# (stage label, input file, start coord label, restrained?) -- do_all_MD order.
STAGES = [
    ("equil1", "md1.inp", "crd", True),
    ("equil2", "md2.inp", "equil1", True),
    ("equil3", "md3.inp", "equil2", True),
    ("equil4", "md4.inp", "equil3", True),
    ("equil5", "md5.inp", "equil4", True),
    ("equil6", "md6.inp", "equil5", True),
    ("heat1", "heat1.in", "equil6", True),
    ("heat2", "heat2.in", "heat1", True),
    ("heat3", "heat3.in", "heat2", True),
    ("heat4", "heat4.in", "heat3", True),
    ("heat5", "heat5.in", "heat4", True),
    ("heat6", "heat6.in", "heat5", True),
    ("equil7", "md7.inp", "heat6", True),
    ("equil8", "md8.inp", "equil7", True),
    ("equil9", "md9.inp", "equil8", True),
    ("equil10", "md10.inp", "equil9", True),
    ("equil11", "md11.inp", "equil10", True),
    ("equil12", "md12.inp", "equil11", True),
    ("prod1", "md13.inp", "equil12", False),
    ("prod2", "md13.inp", "prod1", False),
    ("prod3", "md13.inp", "prod2", False),
    ("prod4", "md13.inp", "prod3", False),
]


def rst(wd, z, label):
    return wd / (f"{z}_complex_wat.crd" if label == "crd" else f"{z}_complex_wat.{label}.rst")


def run_system(z, gpu, log, sysdir=SYS):
    wd = sysdir / z
    top = wd / f"{z}_complex_wat.top"
    if not top.exists():
        log(f"{z}: NO TOPOLOGY, skipping")
        return False
    env, _ = amber_env()
    # PCI bus ordering makes CUDA device indices match nvidia-smi.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    engine = gpu_engine()
    for label, inp, start, restrained in STAGES:
        out_rst = rst(wd, z, label)
        if out_rst.exists() and out_rst.stat().st_size > 0:
            continue
        cin = rst(wd, z, start)
        if not cin.exists():
            log(f"{z}: MISSING start {start} for {label}; abort")
            return False
        base = f"{z}_complex_wat.{label}"
        cmd = [engine, "-O", "-i", str(PROTO / inp), "-p", str(top),
               "-c", str(cin), "-r", str(out_rst),
               "-o", str(wd / f"{base}.out"), "-inf", str(wd / f"{base}.mdinfo")]
        if label.startswith("prod"):
            cmd += ["-x", str(wd / f"{base}.mdcrd")]
        if restrained:
            cmd += ["-ref", str(cin)]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True)
        if p.returncode != 0 or not out_rst.exists():
            log(f"{z}: FAILED at {label} (gpu {gpu}) rc={p.returncode}: "
                f"{(p.stderr or '')[-200:]}")
            return False
        log(f"{z}: {label} done gpu{gpu} {time.time()-t0:.0f}s")
    # Remove equilibration trajectories unless KEEP_EQUIL_MDCRD=1; production
    # trajectories are kept.
    if os.environ.get("KEEP_EQUIL_MDCRD", "0") != "1":
        for f in wd.glob(f"{z}_complex_wat.equil*.mdcrd"):
            f.unlink(missing_ok=True)
    log(f"{z}: COMPLETE (prod4 ready)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=PDB_IDS)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--sysdir", default=str(SYS),
                    help="dir with per-system topologies (use a fresh dir for replicas)")
    ap.add_argument("--log", default=str(REPO / "md" / "md_run.log"))
    args = ap.parse_args()
    # pmemd runs with cwd set to each system directory, so the path must be absolute.
    sysdir = Path(args.sysdir).resolve()
    logf = open(args.log, "a", buffering=1)
    lock = threading.Lock()

    def log(msg):
        with lock:
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            logf.write(line + "\n")

    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    work_q = queue.Queue()
    for z in args.systems:
        work_q.put(z)

    def worker():
        while True:
            try:
                z = work_q.get_nowait()
            except queue.Empty:
                return
            gpu = gpu_q.get()
            try:
                run_system(z, gpu, log, sysdir=sysdir)
            finally:
                gpu_q.put(gpu)
                work_q.task_done()

    threads = [threading.Thread(target=worker) for _ in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("ALL SYSTEMS DONE")


if __name__ == "__main__":
    main()
