#!/usr/bin/env python3
"""Correlation with experiment of GFN2-xTB/CPCM-X scores from score_cpcmx.py.

For each frame set, compares CPCM-X, CPCM-X without the CDS term, and GFN2-xTB with
GBSA, ALPB and ddCOSMO on the same frames. Frames are combined per trajectory with
unweighted, MD-population, 298 K Boltzmann or scatter-scaled basin weights. Reports
R2 and Spearman rho of the five-trajectory pool, min/mean/max R2 over three-trajectory
pools and per-trajectory R2. Reads existing JSON results and writes a metrics CSV.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA, WORK

REPO = WORK
SCR = REPO / "screening_ff_md_sqm"
V2 = SCR / "_xtb_cpcmx_v2_propka"
REPS = ("r1", "r2", "r3", "r4", "r5")
KT = 0.00198720425864083 * 298.15
XTB_ROOT = {
    "iface": SCR / "_xtb_propka_clust_iface_sc_gt5",
    "aa": SCR / "_xtb_propka_clust_aa_gt5",
    "bb": SCR / "_xtb_propka_clust_bb_gt5",
    "contacts": SCR / "_xtb_propka_clust_contacts_gt5",
    "even10": SCR / "_xtb_propka",
}
SNAP = {
    "iface": REPO / "screening_ff_md_sqm_propka_clust_iface_sc_gt5",
    "aa": REPO / "screening_ff_md_sqm_propka_clust_aa_gt5",
    "bb": REPO / "screening_ff_md_sqm_propka_clust_bb_gt5",
    "contacts": REPO / "screening_ff_md_sqm_propka_clust_contacts",
}
COMPARATORS = (("gfn2_gbsa", "GFN2-xTB/GBSA"), ("gfn2_alpb", "GFN2-xTB/ALPB"),
               ("gfn2_cosmo", "GFN2-xTB/ddCOSMO"))


def experimental():
    return (pd.read_csv(DATA / "experimental_data.csv")
              .set_index("PDB_ID")["Exp_dG"].astype(float))


def basin_weights(set_name):
    out = {}
    snap = SNAP.get(set_name)
    if snap is None or not snap.exists():
        return out
    for bf in snap.glob("*/ff14sb/*/basins.json"):
        d = json.loads(bf.read_text())
        for b in d.get("basins", []):
            out[(d["system"], d["rep"], int(b["snap"]))] = float(b["frac"])
    return out


def load_frames(set_name):
    rows = []
    for rep in REPS:
        for jf in (V2 / set_name / rep).rglob("gfn2_cpcmx.json"):
            d = json.loads(jf.read_text())
            rows.append({"rep": rep, "sys": d["system"], "frame": int(d["frame"]),
                         "status": d.get("status"), "cpcmx": d.get("energy"),
                         "cds": d.get("dG_CDS")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for v, _ in COMPARATORS:
        vals = []
        for rep, sys_, fr in zip(df.rep, df.sys, df.frame):
            f = XTB_ROOT[set_name] / rep / sys_ / f"f{fr}" / f"{v}.json"
            e = None
            if f.exists():
                d = json.loads(f.read_text())
                if d.get("status") in ("ok", "cached"):
                    e = d.get("energy")
            vals.append(e)
        df[v] = vals
    w = basin_weights(set_name)
    df["frac"] = [w.get((s, r, f), np.nan) for s, r, f in zip(df.sys, df.rep, df.frame)]
    df["cpcmx_nocds"] = df.cpcmx - df.cds
    return df


def r2_rho(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return np.nan, np.nan
    return (float(stats.pearsonr(x[ok], y[ok])[0] ** 2),
            float(stats.spearmanr(x[ok], y[ok])[0]))


def combine(df, col, how):
    mats, neffs = {}, []
    for rep in REPS:
        vals = {}
        for s, g in df[df.rep == rep].groupby("sys"):
            g = g.dropna(subset=[col])
            if g.empty:
                continue
            e = g[col].to_numpy(float)
            if how == "mean":
                w = np.ones_like(e)
            elif how == "pop":
                w = g.frac.to_numpy(float)
                if not np.isfinite(w).all() or w.sum() <= 0:
                    continue
            elif how == "boltz":
                w = np.exp(-(e - e.min()) / KT)
            elif how == "scatter":
                sd = float(e.std(ddof=1)) if len(e) > 1 else 0.0
                w = np.exp(-(e - e.min()) / sd) if sd > 1e-8 else np.ones_like(e)
            else:
                raise ValueError(how)
            p = w / w.sum()
            vals[s] = float(p @ e)
            neffs.append(float(1.0 / (p @ p)))
        mats[rep] = pd.Series(vals)
    return pd.DataFrame(mats), (float(np.median(neffs)) if neffs else np.nan)


def metrics(mat, exp):
    mat = mat.reindex(exp.index)
    y = exp.to_numpy(float)
    reps = [r for r in REPS if r in mat.columns]
    per = [r2_rho(mat[r].to_numpy(float), y)[0] for r in reps]
    p5, rho = r2_rho(mat[reps].mean(axis=1).to_numpy(float), y)
    p3 = [r2_rho(mat[list(c)].mean(axis=1).to_numpy(float), y)[0]
          for c in combinations(reps, 3)] if len(reps) >= 3 else []
    return {
        "n_sys": int(mat[reps].mean(axis=1).notna().sum()),
        "pool5_r2": p5, "pool5_rho": rho,
        "pool3_min": min(p3) if p3 else np.nan,
        "pool3_mean": float(np.mean(p3)) if p3 else np.nan,
        "pool3_max": max(p3) if p3 else np.nan,
        "spread": (max(per) - min(per)) if len(per) >= 2 else np.nan,
        "per_rep": " / ".join(f"{v:.3f}" for v in per),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["iface", "aa", "bb", "contacts", "even10"])
    args = ap.parse_args()
    exp = experimental()
    rows = []
    for set_name in args.sets:
        df = load_frames(set_name)
        if df.empty:
            print(f"[{set_name}] no frames yet")
            continue
        expected = sum(1 for rep in REPS
                       for _ in (XTB_ROOT[set_name] / rep).rglob("gfn2_gbsa.json"))
        ok = df[df.status == "ok"]
        print(f"\n[{set_name}] CPCM-X frames {len(df)} (ok {len(ok)}), GBSA frames {expected}")
        bad = df[df.status != "ok"]
        if len(bad):
            print(bad[["rep", "sys", "frame", "status"]].to_string(index=False))
        complete = len(ok) == expected and expected > 0
        if not complete:
            print("  incomplete; statistics below are provisional")
        same = ok.dropna(subset=["gfn2_gbsa"])
        hows = ("mean", "pop", "boltz", "scatter") if set_name in SNAP else ("mean", "boltz", "scatter")
        arms = [("cpcmx", "GFN2-xTB/CPCM-X"), ("cpcmx_nocds", "GFN2-xTB/CPCM-X, CDS removed")]
        arms += list(COMPARATORS)
        for col, label in arms:
            for how in hows:
                mat, neff = combine(same, col, how)
                m = metrics(mat, exp)
                rows.append({"set": set_name, "method": label, "weight": how,
                             "complete": complete, "n_frames": int(same[col].notna().sum()),
                             "neff": neff, **m})
                print(f"  {label:30s} {how:8s} pool5 {m['pool5_r2']:.3f} rho {m['pool5_rho']:.3f} "
                      f"spread {m['spread']:.3f} pool3 {m['pool3_min']:.3f}/{m['pool3_mean']:.3f}/"
                      f"{m['pool3_max']:.3f} Neff {neff:.2f}  per {m['per_rep']}")
        d = (same.cpcmx - same.gfn2_gbsa)
        dev = (d - d.groupby(same.sys).transform("median")).abs()
        print(f"  CPCM-X - GBSA per frame: median {d.median():.1f}, "
              f"frames >100 kcal from their system median: {int((dev > 100).sum())}, "
              f"max {dev.max():.1f}")
    out = DATA / "cpcmx_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
