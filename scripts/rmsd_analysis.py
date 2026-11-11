#!/usr/bin/env python3
"""Backbone RMSD for the 100 production trajectories (20 complexes x 5 trajectories).

One cpptraj run per complex and trajectory: autoimage, strip solvent and ions, then
RMSD of the backbone atoms to the first production frame. Per-trajectory summaries
are written to rmsd_analysis.csv.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, WORK  # noqa: E402

REPO = WORK
from amber_setup import amber_bin, amber_env  # noqa

ART = REPO / "md"
REPS = ["r1", "r2", "r3", "r4", "r5"]
PRODS = ("prod1", "prod2", "prod3", "prod4")
OUT = DATA / "rmsd_analysis.csv"
STRIDE = 10          # 400 points per trajectory


def one(args):
    system, rep = args
    d = ART / f"systems_propka_{rep}" / system
    top = d / f"{system}_complex_wat.top"
    trajs = [d / f"{system}_complex_wat.{p}.mdcrd" for p in PRODS]
    if not top.exists() or not all(t.exists() for t in trajs):
        return None
    work = REPO / "_rmsd_tmp" / f"{system}_{rep}"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "rmsd.dat"
    script = "".join(f"trajin {t} 1 last {STRIDE}\n" for t in trajs)
    script += (
        "autoimage\n"
        "strip :WAT,HOH,Na+,Cl-\n"
        "reference 1\n"
        f"rms BB @CA,C,N reference out {out} mass\n"
        "run\nquit\n")
    (work / "in.cpptraj").write_text(script)
    env, _ = amber_env()
    r = subprocess.run([str(amber_bin("cpptraj")), "-p", str(top), "-i", str(work / "in.cpptraj")],
                       capture_output=True, text=True, env=env, cwd=work)
    if not out.exists():
        return {"system": system, "rep": rep, "error": (r.stderr or "")[-200:]}
    vals = []
    for ln in out.read_text().splitlines():
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    if not vals:
        return None
    v = np.array(vals)
    # drop the first 10% as the trajectory settles away from the reference frame
    tail = v[int(0.1 * len(v)):]
    return {"system": system, "rep": rep, "n_frames": len(v),
            "rmsd_mean": float(tail.mean()), "rmsd_sd": float(tail.std(ddof=1)),
            "rmsd_max": float(v.max()), "rmsd_final": float(v[-1])}


def main():
    systems = sorted(p.name for p in (ART / "systems_propka_r1").iterdir() if p.is_dir())
    jobs = [(s, r) for s in systems for r in REPS]
    print(f"{len(jobs)} trajectories")
    rows = []
    with ProcessPoolExecutor(max_workers=20) as pool:
        for i, res in enumerate(pool.map(one, jobs), 1):
            if res:
                rows.append(res)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame([r for r in rows if "error" not in r])
    df.to_csv(OUT, index=False)
    bad = [r for r in rows if "error" in r]
    print(f"wrote {OUT}: {len(df)} trajectories")
    if bad:
        print(f"  {len(bad)} failed, first: {bad[0]}")
    if not df.empty:
        print(f"  backbone RMSD mean {df.rmsd_mean.mean():.2f} A, "
              f"range {df.rmsd_mean.min():.2f}-{df.rmsd_mean.max():.2f}")


if __name__ == "__main__":
    main()
