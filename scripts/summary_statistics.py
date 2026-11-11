#!/usr/bin/env python3
"""Summary statistics over all scoring arms, printed as a sectioned report.

Reads protocol_metrics.csv, protocol_scores.csv and the basin clustering outputs, and
reports method rankings, reference and control arms, clustering, weighting,
solvation and geometry comparisons, the MM/GBSA water series, inter-arm
correlations, ddCOSMO offsets, clustering coverage, outlier residuals and
replica spread extremes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA, WORK

REPO = WORK
REPS = ["r1", "r2", "r3", "r4", "r5"]
m = pd.read_csv(DATA / "protocol_metrics.csv").set_index("arm")
sc = pd.read_csv(DATA / "protocol_scores.csv").set_index("System")
exp = sc["Exp_dG"].astype(float)


def row(arm):
    if arm not in m.index:
        return None
    r = m.loc[arm]
    return (f"R2={r.pool5_r2:.3f} rho={r.pool5_rho:.3f} spread={r.rep_spread:.3f} "
            f"pool3sd={r.pool3_sd:.3f} per_rep=[{r.per_rep_r2}]")


def show(title, arms):
    print(f"\n### {title}")
    for label, a in arms:
        r = row(a)
        print(f"  {label:38s} {r if r else 'MISSING: ' + a}")


print("=" * 78)
print("SUMMARY STATISTICS")
print("=" * 78)
print(f"arms={len(m)}  complexes={len(exp)}  replicas={len(REPS)}")

show("1. Method ranking, top of table", [
    (n, n) for n in m.nlargest(8, "pool5_r2").index])

show("2. Reference and negative controls", [
    ("MM/GBSA Nwat30 (reference)", "MMGBSA_Nwat30"),
    ("MM/GBSA Nwat20", "MMGBSA_Nwat20"),
    ("GFN2/GBSA even-10", "GFN2_GBSA_even10"),
    ("GFN2/ALPB even-10", "GFN2_ALPB_even10"),
    ("GFN1/ALPB even-10", "GFN1_ALPB_even10"),
    ("GFN1/GBSA even-10", "GFN1_GBSA_even10"),
    ("GFN1/ddCOSMO even-10", "GFN1_COSMO_even10"),
    ("PM6-D3H4/COSMO even-10", "PM6_D3H4_COSMO_NOFC_even10"),
])

show("3. H4 implementation trio (same Hamiltonian/solvent/geometry)", [
    ("cuby4-side, charge scaling OFF", "PM6_D3H4_COSMO_NOFC_even10"),
    ("cuby4-side, charge scaling ON", "PM6_D3H4_COSMO_CUBY_even10"),
    ("MOPAC built-in d3h4", "PM6_D3H4_COSMO_even10"),
])

show("4. Clustering schemes at fixed GFN2/GBSA (unweighted)", [
    (f"basin_{t}", f"GFN2_GBSA_basin_{t}_unw") for t in ("aa", "bb", "iface_sc", "contacts")
] + [("even-10 (no clustering)", "GFN2_GBSA_even10")])

show("5. Clustering schemes at fixed PM6-D3H4/COSMO", [
    (f"basin_{t}", f"PM6_D3H4_COSMO_NOFC_basin_{t}_unw")
    for t in ("aa", "bb", "iface_sc", "contacts")
] + [("even-10", "PM6_D3H4_COSMO_NOFC_even10")])

show("6. Population weighting (does it matter?)", [
    (f"{t} unweighted", f"GFN2_GBSA_basin_{t}_unw") for t in ("aa", "iface_sc")
] + [(f"{t} weighted", f"GFN2_GBSA_basin_{t}_wt") for t in ("aa", "iface_sc")])

show("7. Solvation class within a fixed Hamiltonian", [
    ("GFN2/GBSA", "GFN2_GBSA_even10"),
    ("GFN2/ALPB", "GFN2_ALPB_even10"),
    ("GFN2/ddCOSMO (polar only)", "GFN2_COSMO_even10"),
    ("GFN1/GBSA", "GFN1_GBSA_even10"),
    ("GFN1/ALPB", "GFN1_ALPB_even10"),
    ("GFN1/ddCOSMO", "GFN1_COSMO_even10"),
])

show("8. Geometry treatment on aa basins, same preset", [
    ("raw single point", "PM6_D3H4_COSMO_NOFC_basin_aa_unw"),
    ("MM/GB minimised then SP", "PM6_NOFC_basin_aa_mmsp"),
])

show("9. MM/GBSA water series", [
    (f"Nwat={n}", f"MMGBSA_Nwat{n}") for n in (0, 10, 20, 30, 40)])

print("\n### 10. Best PM6/PM7 presets (even-10)")
p = m[[i.endswith("_even10") and i.startswith(("PM6", "PM7")) for i in m.index]]
for a, r in p.nlargest(8, "pool5_r2").iterrows():
    print(f"  {a:38s} R2={r.pool5_r2:.3f} rho={r.pool5_rho:.3f} spread={r.rep_spread:.3f}")

# Inter-arm correlations
print("\n### 11. Inter-arm correlations (pool-5 scores, n=20)")
pairs = [("GFN2_GBSA_even10", "GFN2_ALPB_even10"),
         ("GFN2_GBSA_even10", "GFN2_COSMO_even10"),
         ("GFN2_ALPB_even10", "GFN2_COSMO_even10"),
         ("GFN1_ALPB_even10", "GFN2_ALPB_even10"),
         ("GFN2_GBSA_basin_iface_sc_unw", "GFN2_GBSA_basin_aa_wt"),
         ("GFN2_GBSA_basin_iface_sc_unw", "MMGBSA_Nwat30")]
for a, b in pairs:
    if a in sc.columns and b in sc.columns:
        d = sc[[a, b]].dropna()
        print(f"  r({a}, {b}) = {np.corrcoef(d[a], d[b])[0,1]:.3f}")

# ddCOSMO offset relative to GBSA and ALPB
print("\n### 12. ddCOSMO offset from GBSA and ALPB (kcal/mol, per complex)")
if {"GFN2_COSMO_even10", "GFN2_ALPB_even10"} <= set(sc.columns):
    for ref in ("GFN2_ALPB_even10", "GFN2_GBSA_even10"):
        d = (sc["GFN2_COSMO_even10"] - sc[ref]).dropna()
        print(f"  vs {ref}: mean={d.mean():+.1f} median={d.median():+.1f} "
              f"range={d.min():+.1f}..{d.max():+.1f}  n_overbound={int((d<0).sum())}/{len(d)}")

# Clustering coverage and basin counts
print("\n### 13. Clustering coverage and basin counts (mean per system-replica)")
for tag in ("aa", "bb", "iface_sc", "contacts"):
    base = REPO / f"screening_ff_md_sqm_propka_clust_{tag}_gt5"
    if not base.exists():
        base = REPO / f"screening_ff_md_sqm_propka_clust_{tag}"
    cov, nb, k = [], [], 0
    for bf in base.glob("*/ff14sb/*/basins.json"):
        if bf.parent.name not in REPS:
            continue
        d = json.loads(bf.read_text())
        cov.append(d.get("covered_frac", np.nan))
        nb.append(d.get("n_basins", np.nan))
        k += 1
    if k:
        print(f"  {tag:9s} cells={k:3d} coverage={100*np.nanmean(cov):.1f}% "
              f"basins/cell={np.nanmean(nb):.2f}")

# Outlier structure
print("\n### 14. Outlier structure: standardised residuals across main arms")
main = ["MMGBSA_Nwat30", "PM6_D3H4_COSMO_NOFC_even10", "GFN2_ALPB_even10",
        "GFN2_GBSA_even10", "GFN2_GBSA_basin_aa_unw", "GFN2_GBSA_basin_iface_sc_unw"]
main = [a for a in main if a in sc.columns]
Z, RAW = {}, {}
for a in main:
    d = sc[[a]].join(exp).dropna()
    x, y = d[a].to_numpy(float), d["Exp_dG"].to_numpy(float)
    sl, ic = np.polyfit(x, y, 1)
    res = y - (sl * x + ic)
    Z[a] = pd.Series(res / res.std(ddof=1), index=d.index)
    RAW[a] = pd.Series(res, index=d.index)
z = pd.DataFrame(Z); raw = pd.DataFrame(RAW)
mz = z.abs().mean(axis=1).sort_values(ascending=False)
print(f"  arms used: {len(main)}")
for s in mz.head(6).index:
    print(f"  {s}: mean|z|={mz[s]:.2f} signed residual {raw.loc[s].min():+.2f}..{raw.loc[s].max():+.2f} "
          f"kcal/mol, |z|>1.5 in {int((z.loc[s].abs()>1.5).sum())}/{len(main)} arms")

print("\n### 15. Replica spread extremes across all complete arms")
comp = m[m.n_sys == 20]
print(f"  tightest: {comp.rep_spread.idxmin()} {comp.rep_spread.min():.3f}")
print(f"  widest:   {comp.rep_spread.idxmax()} {comp.rep_spread.max():.3f}")
print(f"  MMGBSA family spread range: "
      f"{comp[[i.startswith('MMGBSA') for i in comp.index]].rep_spread.min():.3f}"
      f"-{comp[[i.startswith('MMGBSA') for i in comp.index]].rep_spread.max():.3f}")
