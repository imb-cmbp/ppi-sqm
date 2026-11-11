#!/usr/bin/env python3
"""Selection stability and winner's curse among scoring arms (paired bootstrap).

Arms are the complete columns of protocol_scores.csv plus the GFN2-xTB/CPCM-X arms,
all scored on the same 20 complexes. Over resamples of complexes (fixed seed) this
counts how often each arm and each method family has the highest R2, and estimates
the winner's curse: the mean gap between the maximum R2 on a resample and the R2 of
the full-data best arm on the same resample. Writes ranking_bootstrap_top.csv (top arms),
ranking_bootstrap_family.csv and ranking_bootstrap_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import cpcmx_arms

from paths import DATA

REPO = DATA
OUT = REPO / "ranking_bootstrap_top.csv"
OUT_FAMILY = REPO / "ranking_bootstrap_family.csv"
OUT_SUMMARY = REPO / "ranking_bootstrap_summary.csv"
N_BOOT = 20000
SEED = 20240912
FAMILIES = [("GFN2", "GFN2-xTB"), ("GFN1", "GFN1-xTB"), ("PM6", "PM6"),
            ("PM7", "PM7"), ("MMGBSA", "MM/GBSA")]

sc = cpcmx_arms.load_scores(pd.read_csv(REPO / "protocol_scores.csv").set_index("System"))
sc = sc.join(pd.read_csv(REPO / "cpcmx_arms_scores_wt.csv").set_index("System"))
exp = sc["Exp_dG"].astype(float)
arms = [c for c in sc.columns if c != "Exp_dG"]

X = sc[arms].to_numpy(float)          # systems x arms
y = exp.to_numpy(float)
keep = ~np.isnan(X).any(axis=0)       # arms complete on all 20
X, arms = X[:, keep], [a for a, k in zip(arms, keep) if k]
n, k = X.shape
print(f"{n} complexes x {k} complete arms")


def family(arm: str) -> str:
    return arm.split("_")[0]


def r2_cols(Xs, ys):
    """R2 of every column against ys, vectorised."""
    Xc = Xs - Xs.mean(axis=0)
    yc = ys - ys.mean()
    num = (Xc * yc[:, None]).sum(axis=0) ** 2
    den = (Xc ** 2).sum(axis=0) * (yc ** 2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


full = r2_cols(X, y)
order = np.argsort(-full)
best_i = int(order[0])
print(f"nominal best: {arms[best_i]}  R2={full[best_i]:.3f}")

is_cpx = np.array([a.startswith("GFN2_CPCMX") for a in arms])
i_1emv = list(sc.index).index("1EMV")    # largest CPCM-X outlier

rng = np.random.default_rng(SEED)
wins = np.zeros(k, dtype=int)
max_minus_nominal = []
nominal_vals = []
has_1emv, cpx_won = [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    if len(np.unique(idx)) < 3:
        continue
    r = r2_cols(X[idx], y[idx])
    if np.all(np.isnan(r)):
        continue
    w = int(np.nanargmax(r))
    wins[w] += 1
    max_minus_nominal.append(np.nanmax(r) - r[best_i])
    nominal_vals.append(r[best_i])
    has_1emv.append(i_1emv in idx)
    cpx_won.append(bool(is_cpx[w]))

tot = wins.sum()
max_minus_nominal = np.array(max_minus_nominal)
nominal_vals = np.array(nominal_vals)

print(f"\nresamples used: {tot}")
print("\n=== selection stability: how often each arm is the winner ===")
rows = []
for i in np.argsort(-wins)[:10]:
    frac = wins[i] / tot
    rows.append({"arm": arms[i], "full_r2": round(float(full[i]), 4),
                 "win_fraction": round(float(frac), 4)})
    print(f"  {arms[i]:34s} full R2={full[i]:.3f}  wins {100*frac:5.1f}% of resamples")

nominal_share = wins[best_i] / tot
print(f"\n  NOMINAL BEST ARM wins {100*nominal_share:.1f}% of resamples "
      f"(rank by wins: {int((wins > wins[best_i]).sum()) + 1} of {k})")

print("\n=== wins by method family ===")
fam_rows = []
for key, name in FAMILIES:
    members = [i for i, a in enumerate(arms) if family(a) == key]
    share = wins[members].sum() / tot
    fam_rows.append({"family": name, "n_arms": len(members),
                     "win_fraction": round(float(share), 4),
                     "best_full_r2": round(float(full[members].max()), 4)})
    print(f"  {name:10s} {len(members):3d} arms  wins {100*share:5.1f}%  "
          f"best full R2 {full[members].max():.3f}")
cpx = [i for i, a in enumerate(arms) if a.startswith("GFN2_CPCMX")]
cpx_share = wins[cpx].sum() / tot
print(f"  of which GFN2-xTB/CPCM-X ({len(cpx)} arms) wins {100*cpx_share:.1f}%")
has_1emv, cpx_won = np.array(has_1emv), np.array(cpx_won)
cpx_with = cpx_won[has_1emv].mean()
cpx_without = cpx_won[~has_1emv].mean()
print(f"  CPCM-X wins {100*cpx_with:.1f}% of resamples containing 1EMV and "
      f"{100*cpx_without:.1f}% of the {100*(~has_1emv).mean():.1f}% without it")

print("\n=== winner's curse ===")
print(f"  mean(max R2 on resample - nominal-best R2 on same resample) "
      f"= {max_minus_nominal.mean():+.3f}")
print(f"  nominal best arm, bootstrap mean R2 = {nominal_vals.mean():.3f} "
      f"[{np.percentile(nominal_vals,2.5):.3f}, {np.percentile(nominal_vals,97.5):.3f}]")

within = int((full >= full[best_i] - 0.02).sum())
print(f"\n  arms within 0.02 of the best: {within}")

pd.DataFrame(rows).to_csv(OUT, index=False)
pd.DataFrame(fam_rows).to_csv(OUT_FAMILY, index=False)
pd.Series({
    "n_arms": k, "n_resamples": int(tot),
    "nominal_best_arm": arms[best_i], "nominal_best_r2": float(full[best_i]),
    "nominal_best_share": float(nominal_share),
    "top_share": float(wins.max() / tot),
    "optimism": float(max_minus_nominal.mean()),
    "n_within_002": within,
    "cpcmx_share": float(cpx_share), "cpcmx_n_arms": len(cpx),
    "cpcmx_best_full_r2": float(full[cpx].max()),
    "cpcmx_share_with_1emv": float(cpx_with),
    "cpcmx_share_without_1emv": float(cpx_without),
    "share_without_1emv": float((~has_1emv).mean()),
}, name="value").rename_axis("quantity").to_csv(OUT_SUMMARY)
print(f"\nwrote {OUT.name}, {OUT_FAMILY.name}, {OUT_SUMMARY.name}")
