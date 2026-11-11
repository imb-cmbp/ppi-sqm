#!/usr/bin/env python3
"""Collect the per-complex, per-trajectory scores of every protocol from the
calculation tree and summarise them.

Writes protocol_scores.csv (score per complex, averaged over the five
trajectories) and protocol_metrics.csv (pooled, per-trajectory and
three-trajectory R2, Spearman rho and leave-one-out Q2 per protocol).

Protocols:
  even-10   20 PM6/PM7 presets, 6 xTB variants, MM/GBSA at 5 water counts
  basins    the same presets and variants under the four clustering schemes,
            unweighted and population-weighted
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA, WORK

REPO = WORK
SCR = REPO / "screening_ff_md_sqm"
ART = REPO / "md"
REPS = ["r1", "r2", "r3", "r4", "r5"]
FF = "ff14sb"

NWATS = (0, 10, 20, 30, 40)
XTB_VARIANTS = ("gfn1_alpb", "gfn1_cosmo", "gfn1_gbsa",
                "gfn2_alpb", "gfn2_cosmo", "gfn2_gbsa")
SCHEMES = ("aa", "bb", "iface_sc", "contacts")


def experimental():
    df = pd.read_csv(DATA / "experimental_data.csv")
    return df.rename(columns={"PDB_ID": "System"})[["System", "Exp_dG"]]


def r2(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return np.nan
    return stats.pearsonr(x[ok], y[ok])[0] ** 2


def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return np.nan
    return stats.spearmanr(x[ok], y[ok])[0]


def loo_q2(X, y):
    """Leave-one-out Q2 for OLS of y on [1|X]."""
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[ok], y[ok]
    n = len(y)
    if n < 6:
        return np.nan
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        A = np.column_stack([np.ones(m.sum()), X[m]])
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred[i] = np.r_[1.0, X[i]] @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def metrics(name, mat, exp):
    """Per-trajectory, three-trajectory and pooled metrics for one
    complex x trajectory score matrix."""
    if mat is None or mat.empty:
        return None
    mat = mat.reindex(exp.index)
    reps = [c for c in REPS if c in mat.columns]
    y = exp.values.astype(float)
    per = {c: r2(mat[c].values.astype(float), y) for c in reps}
    pool3 = []
    if len(reps) >= 3:
        for sub in combinations(reps, 3):
            pool3.append(r2(mat[list(sub)].mean(axis=1).values.astype(float), y))
    p5 = mat[reps].mean(axis=1).values.astype(float)
    return {
        "arm": name,
        "n_sys": int(np.isfinite(p5).sum()),
        "n_rep": len(reps),
        "pool5_r2": r2(p5, y),
        "pool5_rho": spearman(p5, y),
        "pool5_looq2": loo_q2(p5.reshape(-1, 1), y),
        "pool3_mean": float(np.nanmean(pool3)) if pool3 else np.nan,
        "pool3_sd": float(np.nanstd(pool3)) if pool3 else np.nan,
        "pool3_min": float(np.nanmin(pool3)) if pool3 else np.nan,
        "pool3_max": float(np.nanmax(pool3)) if pool3 else np.nan,
        "per_rep_r2": " / ".join(f"{per[c]:.3f}" if np.isfinite(per[c]) else "-"
                                 for c in reps),
        "rep_spread": (max(per.values()) - min(per.values())
                       if per and all(np.isfinite(v) for v in per.values())
                       else np.nan),
    }


def snap_base(tag):
    """Basin snapshot root. cluster_contacts.py writes its basins directly into
    the cluster directory, so the contact scheme has no separate _gt5 tree."""
    d = REPO / f"screening_ff_md_sqm_propka_clust_{tag}_gt5"
    return d if d.exists() else REPO / f"screening_ff_md_sqm_propka_clust_{tag}"


# Loaders

def mmgbsa_even(nwat):
    """Complex x trajectory Nwat-MM/GBSA dG over 100 frames of the final nanosecond."""
    cols = {}
    for rep in REPS:
        f = ART / f"mmgbsa_results_propka_{rep}_polar.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d["Nwat"] == nwat]
        s = pd.Series(d.set_index("System")["dG"]).dropna()
        if not s.empty:
            cols[rep] = s
    return pd.DataFrame(cols)


def pm6_from_harvest(root, preset):
    """System x replica mean PM6 interaction energy from the harvested CSV.

    The harvest already averages over the frames (even-10) or basin centroids
    (clustering arms) that the scorer completed, so this is the unweighted mean.
    """
    col = f"{preset}_sp_wet"
    cols = {}
    for rep in REPS:
        f = root / rep / f"detailed_{rep}_pm6series.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if col not in d.columns:
            continue
        s = pd.Series(d.set_index("System")[col]).dropna()
        if not s.empty:
            cols[rep] = s
    return pd.DataFrame(cols)


def xtb_from_json(root, variant, weights=None):
    """System x replica xTB interaction energy, mean over frames/basins.

    weights maps (system, rep, snap) -> basin population fraction. When given,
    the per-system value is the population-weighted mean over basins,
    renormalised over the basins that actually scored.
    """
    cols = {}
    for rep in REPS:
        base = root / rep
        if not base.exists():
            continue
        acc = {}
        for jf in base.rglob(f"{variant}.json"):
            try:
                d = json.loads(jf.read_text())
            except (OSError, ValueError):
                continue
            if d.get("status") not in ("ok", "cached") or d.get("energy") is None:
                continue
            system = d.get("system") or jf.parent.parent.name
            snap = d.get("frame")
            if snap is None:
                continue
            w = 1.0
            if weights is not None:
                w = weights.get((system, rep, int(snap)))
                if w is None:
                    continue
            num, den = acc.get(system, (0.0, 0.0))
            acc[system] = (num + w * float(d["energy"]), den + w)
        s = pd.Series({k: n / d for k, (n, d) in acc.items() if d > 0})
        if not s.empty:
            cols[rep] = s
    return pd.DataFrame(cols)


def pm6_mm_sp(root, preset="pm6_d3h4_cosmo_nofc"):
    """MM/GB-minimised geometries; the harvested column carries the mm_sp phase tag."""
    col = f"{preset}_mm_sp_wet"
    cols = {}
    for rep in REPS:
        f = root / rep / f"detailed_{rep}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if col not in d.columns:
            continue
        s = pd.Series(d.set_index("System")[col]).dropna()
        if not s.empty:
            cols[rep] = s
    return pd.DataFrame(cols)


def basin_weights(tag):
    """(system, rep, snap) -> population fraction, from basins.json."""
    out = {}
    base = snap_base(tag)
    for bf in base.glob(f"*/{FF}/*/basins.json"):
        try:
            d = json.loads(bf.read_text())
        except (OSError, ValueError):
            continue
        system, rep = d.get("system"), d.get("rep")
        for b in d.get("basins", []):
            if b.get("snap") is not None and b.get("frac") is not None:
                out[(system, rep, int(b["snap"]))] = float(b["frac"])
    return out


def mmgbsa_basin(tag, kind="unweighted"):
    """System x replica MM/GBSA over basin centroids."""
    f = ART / f"mmgbsa_propka_clust_{tag}_gt5" / f"mmgbsa_{kind}.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    if not {"System", "rep", "dG"} <= set(d.columns):
        return pd.DataFrame()
    return d.pivot_table(index="System", columns="rep", values="dG")


# Protocols

def pm6_presets():
    """Preset names, read from the header of the harvested CSV."""
    for rep in REPS:
        f = SCR / "_qmmm_trunc_propka" / rep / f"detailed_{rep}_pm6series.csv"
        if f.exists():
            cols = pd.read_csv(f, nrows=0).columns
            return [c[: -len("_sp_wet")] for c in cols if c.endswith("_sp_wet")]
    return []


def build_arms():
    """Complex x trajectory score matrices for every protocol, keyed by name."""
    arms = {}

    for n in NWATS:
        arms[f"MMGBSA_Nwat{n}"] = mmgbsa_even(n)

    presets = pm6_presets()
    for p in presets:
        arms[f"{p.upper()}_even10"] = pm6_from_harvest(
            SCR / "_qmmm_trunc_propka", p)
    for v in XTB_VARIANTS:
        arms[f"{v.upper()}_even10"] = xtb_from_json(SCR / "_xtb_propka", v)

    for tag in SCHEMES:
        w = basin_weights(tag)
        qroot = SCR / f"_qmmm_trunc_propka_clust_{tag}_gt5"
        xroot = SCR / f"_xtb_propka_clust_{tag}_gt5"
        for p in presets:
            arms[f"{p.upper()}_basin_{tag}_unw"] = pm6_from_harvest(qroot, p)
        for v in XTB_VARIANTS:
            arms[f"{v.upper()}_basin_{tag}_unw"] = xtb_from_json(xroot, v)
            arms[f"{v.upper()}_basin_{tag}_wt"] = xtb_from_json(xroot, v, weights=w)
        arms[f"MMGBSA_basin_{tag}_unw"] = mmgbsa_basin(tag, "unweighted")
        arms[f"MMGBSA_basin_{tag}_wt"] = mmgbsa_basin(tag, "weighted")

    # MM/GB minimisation before the single point, on the all-atom >5% basins;
    # compare with PM6_D3H4_COSMO_NOFC_basin_aa_unw on the same geometries.
    arms["PM6_NOFC_basin_aa_mmsp"] = pm6_mm_sp(
        SCR / "_qmmm_trunc_propka_clust_aa_gt5_mmsp_igb7")

    return {k: v for k, v in arms.items() if v is not None and not v.empty}


def main():
    exp = experimental().set_index("System")["Exp_dG"].astype(float)
    arms = build_arms()

    rows, score_cols = [], {}
    for name, mat in arms.items():
        m = metrics(name, mat, exp)
        if m:
            rows.append(m)
        aligned = mat.reindex(exp.index)
        score_cols[name] = aligned.mean(axis=1)

    met = pd.DataFrame(rows).sort_values("pool5_r2", ascending=False)
    met.to_csv(DATA / "protocol_metrics.csv", index=False)

    sc = pd.DataFrame(score_cols)
    sc.insert(0, "Exp_dG", exp)
    sc.to_csv(DATA / "protocol_scores.csv")

    print(f"{len(arms)} arms -> protocol_metrics.csv, protocol_scores.csv")
    print(f"{len(exp)} complexes, replicas {', '.join(REPS)}\n")
    show = ["arm", "n_rep", "n_sys", "pool5_r2", "pool5_rho", "rep_spread"]
    show = [c for c in show if c in met.columns]
    print(met[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
