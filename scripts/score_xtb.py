#!/usr/bin/env python3
"""GFN-xTB interaction energies on the capped interface models built by score_mopac.py.

Scores each system/frame model under --shell-base with the variants in VARIANTS
(GFN2 in gas phase or with GBSA, ALPB, ddCOSMO or CPCM-X; GFN1 with GBSA, ALPB or
ddCOSMO) on the same geometries as the MOPAC runs. Results are cached as JSON per
(system, frame, variant) under --out-base; per-system mean, SD and frame count are
written to --out-csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import xtb_runner as X

from paths import WORK

REPO = WORK
DEFAULT_SHELLS = REPO / "screening_ff_md_sqm" / "_qmmm_trunc_propka" / "r1"

# name: (gfn, solvent, solvation_model)
VARIANTS = {
    "gfn2_alpb": (2, "water", "alpb"),
    "gfn2_gbsa": (2, "water", "gbsa"),
    "gfn2_gas": (2, None, None),
    "gfn2_cosmo": (2, "water", "cosmo"),
    "gfn2_cpcmx": (2, "water", "cpcmx"),
    "gfn1_alpb": (1, "water", "alpb"),
    "gfn1_gbsa": (1, "water", "gbsa"),
    "gfn1_cosmo": (1, "water", "cosmo"),
}


def _pdb_ids():
    import score_mopac as S
    return S.PDB_IDS


def run_one(task):
    system, frame, variant, shell_dir, out_dir, threads, accuracy = task
    gfn, solvent, solvation = VARIANTS[variant]
    res_json = Path(out_dir) / system / f"f{frame}" / f"{variant}.json"
    if res_json.exists():
        try:
            d = json.loads(res_json.read_text())
            st = d.get("status", "cached")
            retry_failed = os.environ.get("XTB_RETRY_FAILED") == "1"
            if st in ("ok", "charge_retry_ok") or not retry_failed:
                return system, frame, variant, d.get("energy"), st
        except Exception:  # noqa: BLE001
            pass

    shell = Path(shell_dir)
    capped, meta_f = shell / "capped.pdb", shell / "capped.trunc.json"
    if not capped.exists() or not meta_f.exists():
        return system, frame, variant, None, "missing_shell"
    meta = json.loads(meta_f.read_text())

    work = res_json.parent / f"_work_{variant}_{os.getpid()}"
    t0 = time.time()
    extra = {}
    try:
        # For CPCM-X, also store the interaction differences of CPCMX_SIDECAR_KEYS;
        # energy is E(AB) - E(A) - E(B) of the total energies, as for the other variants.
        if solvation == "cpcmx":
            rec, status = X.interaction_decomp(
                capped, work, meta["charge_a"], meta["charge_b"],
                solvent=solvent, gfn=gfn, threads=threads, accuracy=accuracy,
                solvation=solvation)
            if rec is None:
                energy = None
            else:
                energy = rec["energy"]
                for k in X.CPCMX_SIDECAR_KEYS:
                    if k in rec:
                        extra[k] = rec[k]
        else:
            energy, status = X.interaction_energy(
                capped, work, meta["charge_a"], meta["charge_b"],
                solvent=solvent, gfn=gfn, threads=threads, accuracy=accuracy,
                solvation=solvation or "alpb")
    except Exception as exc:  # noqa: BLE001
        energy, status = None, f"exc:{type(exc).__name__}:{exc}"[:200]
    dt = time.time() - t0

    res_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system, "frame": frame, "variant": variant,
        "energy": energy, "status": status, "seconds": round(dt, 1),
        "n_atoms": meta.get("n_atoms"),
        "charge_a": meta["charge_a"], "charge_b": meta["charge_b"],
    }
    payload.update(extra)
    tmp = res_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    if res_json.exists():
        try:
            prev = json.loads(res_json.read_text())
            if (prev.get("status") in ("ok", "charge_retry_ok")
                    and status not in ("ok", "charge_retry_ok")):
                tmp.unlink(missing_ok=True)
                return system, frame, variant, prev.get("energy"), prev.get("status")
        except Exception:  # noqa: BLE001
            pass
    tmp.replace(res_json)
    for stem in ("a", "b", "ab"):
        src = Path(work) / f"{stem}.xtbout"
        if src.exists():
            try:
                shutil.copy2(src, res_json.parent / f"{variant}_{stem}.xtbout")
            except OSError:
                pass
    X.cleanup(work)
    return system, frame, variant, energy, status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shell-base", default=str(DEFAULT_SHELLS))
    ap.add_argument("--out-base", default=None)
    ap.add_argument("--cutoff", default="c6")
    ap.add_argument("--hydration", default="wet", choices=["wet", "dry"])
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="Frame indices (default 1..10). Ignored with --auto-frames.")
    ap.add_argument("--auto-frames", action="store_true",
                    help="Discover fN/capped.pdb under each system instead of "
                         "assuming frames 1..10 (needed for multi-basin clusters).")
    ap.add_argument("--variants", nargs="*", default=["gfn2_alpb", "gfn2_gas"],
                    choices=sorted(VARIANTS))
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--threads", type=int, default=2,
                    help="OMP threads per xtb job (workers*threads = cores used)")
    ap.add_argument("--accuracy", type=float, default=1.0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    shell_base = Path(args.shell_base)
    out_base = Path(args.out_base) if args.out_base else shell_base.parent / "_xtb_rep3"
    systems = args.systems or _pdb_ids()
    default_frames = args.frames or list(range(1, 11))

    def _frames_for(system):
        if not args.auto_frames:
            return default_frames
        d = shell_base / system / args.cutoff / args.hydration
        found = []
        if d.is_dir():
            for p in d.glob("f*"):
                if (p / "capped.pdb").exists():
                    try:
                        found.append(int(p.name[1:]))
                    except ValueError:
                        pass
        return sorted(found)

    tasks = []
    for s in systems:
        for f in _frames_for(s):
            sd = shell_base / s / args.cutoff / args.hydration / f"f{f}"
            for v in args.variants:
                tasks.append((s, f, v, str(sd), str(out_base), args.threads, args.accuracy))

    print(f"xTB scoring: {len(systems)} systems, {len(tasks)} jobs "
          f"({len(args.variants)} variants, auto_frames={args.auto_frames})")
    print(f"shells: {shell_base}\nout:    {out_base}")
    print(f"workers: {args.workers} x {args.threads} threads = "
          f"{args.workers * args.threads} cores")

    results = {}
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            s, f, v, e, st = fut.result()
            results.setdefault((s, v), {})[f] = (e, st)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - done) / max(rate, 1e-9) / 60
                print(f"  [{done}/{len(tasks)}] {s} f{f} {v}: "
                      f"{'NaN' if e is None else f'{e:.1f}'} ({st}) ETA {eta:.0f} min",
                      flush=True)

    out_csv = Path(args.out_csv) if args.out_csv else out_base / "harvested_xtb.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["System"]
    for v in args.variants:
        cols += [f"{v}_sp_{args.hydration}", f"{v}_sp_{args.hydration}_std",
                 f"{v}_sp_{args.hydration}_n"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in systems:
            row = {"System": s}
            for v in args.variants:
                vals = [e for e, _ in results.get((s, v), {}).values() if e is not None]
                base = f"{v}_sp_{args.hydration}"
                row[base] = round(float(np.mean(vals)), 4) if vals else ""
                row[f"{base}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else ""
                row[f"{base}_n"] = len(vals)
            w.writerow(row)

    print(f"\nWrote {out_csv}")
    for v in args.variants:
        ok = sum(1 for (s, vv), d in results.items() if vv == v
                 for e, _ in d.values() if e is not None)
        print(f"  {v}: {ok}/{len(tasks) // max(len(args.variants), 1)} frames succeeded")
    fails = {st for d in results.values() for _, st in d.values()
             if st not in ("ok", "cached", "charge_retry_ok")}
    if fails:
        print(f"  failure modes: {sorted(fails)[:5]}")


if __name__ == "__main__":
    os.nice(10)
    main()
