#!/usr/bin/env python3
"""Compare two scoring arms by their correlation with the same experimental Exp_dG.

Williams (1959) t test for dependent correlations sharing one criterion (the arms
score the same complexes and correlate with each other, so an independent-sample
Fisher-z test does not apply); a seeded paired bootstrap over complexes giving a
CI on the R^2 difference without a normality assumption; and the sample size at
which the observed gap reaches 80% power. R^2 is the squared Pearson correlation.

Usage:
    python3 compare_methods.py --scores protocol_scores.csv --arm-a GFN2_ALPB_even10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA

SCORES = DATA / "protocol_scores.csv"
OUT = DATA / "method_comparison.csv"

# Default pair: an SQM arm and the Nwat-MM/GBSA reference protocol.
DEFAULT_A = "GFN2_GBSA_basin_iface_sc_unw"
DEFAULT_B = "MMGBSA_Nwat30"

N_BOOT = 20000
SEED = 20240912
POWER = 0.80
ALPHA = 0.05


def williams_t(r_ay: float, r_by: float, r_ab: float, n: int) -> tuple[float, float, int]:
    """Williams test for two dependent correlations sharing a criterion y.

    r_ay  correlation of arm A with experiment
    r_by  correlation of arm B with experiment
    r_ab  correlation of the two arms with each other
    n     number of paired observations

    Returns (t, two-sided p, df). df = n - 3.
    """
    df = n - 3
    det = (1.0 - r_ay**2 - r_by**2 - r_ab**2) + 2.0 * r_ay * r_by * r_ab
    rbar = 0.5 * (r_ay + r_by)
    # The denominator already includes the leading factor of 2 in Williams'
    # expression; it is not applied again in the ratio.
    denom = (2.0 * (n - 1.0) / (n - 3.0)) * det + (rbar**2) * ((1.0 - r_ab) ** 3)
    t = (r_ay - r_by) * np.sqrt(((n - 1.0) * (1.0 + r_ab)) / denom)
    p = 2.0 * stats.t.sf(abs(t), df)
    return float(t), float(p), int(df)


def required_n(r_ay: float, r_by: float, r_ab: float,
               power: float = POWER, alpha: float = ALPHA,
               n_max: int = 200000) -> int | None:
    """Smallest n at which the observed correlation gap reaches `power`.

    Holds the three observed correlations fixed and increases n until the Williams
    statistic clears the two-sided critical value with the requested power. Returns
    None if the gap never becomes resolvable within n_max.
    """
    for n in range(10, n_max):
        t, _, df = williams_t(r_ay, r_by, r_ab, n)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        # Power for a noncentral-t with noncentrality equal to the observed statistic.
        pw = stats.nct.sf(t_crit, df, abs(t)) + stats.nct.cdf(-t_crit, df, abs(t))
        if pw >= power:
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", default=DEFAULT_A)
    ap.add_argument("--arm-b", default=DEFAULT_B)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    scores, out_path = Path(args.scores), Path(args.out)
    d = pd.read_csv(scores)
    for col in (args.arm_a, args.arm_b):
        if col not in d.columns:
            raise SystemExit(f"arm not in {scores.name}: {col}")

    sub = d[["System", "Exp_dG", args.arm_a, args.arm_b]].dropna()
    n = len(sub)
    y = sub["Exp_dG"].to_numpy(float)
    a = sub[args.arm_a].to_numpy(float)
    b = sub[args.arm_b].to_numpy(float)

    r_ay = float(np.corrcoef(a, y)[0, 1])
    r_by = float(np.corrcoef(b, y)[0, 1])
    r_ab = float(np.corrcoef(a, b)[0, 1])
    t, p, df = williams_t(r_ay, r_by, r_ab, n)

    rho_a = float(stats.spearmanr(a, y).statistic)
    rho_b = float(stats.spearmanr(b, y).statistic)

    # Paired bootstrap over complexes. Resamples with fewer than 3 distinct systems,
    # or with an undefined correlation, are skipped.
    rng = np.random.default_rng(args.seed)
    da, db = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        ys = y[idx]
        if np.std(ys) == 0:
            continue
        ra = np.corrcoef(a[idx], ys)[0, 1]
        rb = np.corrcoef(b[idx], ys)[0, 1]
        if np.isnan(ra) or np.isnan(rb):
            continue
        da.append(ra**2)
        db.append(rb**2)
    da, db = np.asarray(da), np.asarray(db)
    diff = da - db

    need = required_n(r_ay, r_by, r_ab)

    rows = [
        ("arm_a", args.arm_a),
        ("arm_b", args.arm_b),
        ("n_systems", n),
        ("r_a_exp", round(r_ay, 4)),
        ("r_b_exp", round(r_by, 4)),
        ("r_a_b", round(r_ab, 4)),
        ("r2_a", round(r_ay**2, 4)),
        ("r2_b", round(r_by**2, 4)),
        ("delta_r2", round(r_ay**2 - r_by**2, 4)),
        ("spearman_a", round(rho_a, 4)),
        ("spearman_b", round(rho_b, 4)),
        ("williams_t", round(t, 4)),
        ("williams_df", df),
        ("williams_p", round(p, 4)),
        ("boot_n_effective", len(diff)),
        ("boot_delta_r2_mean", round(float(diff.mean()), 4)),
        ("boot_delta_r2_lo95", round(float(np.percentile(diff, 2.5)), 4)),
        ("boot_delta_r2_hi95", round(float(np.percentile(diff, 97.5)), 4)),
        ("boot_p_a_better", round(float((diff > 0).mean()), 4)),
        ("boot_r2_a_lo95", round(float(np.percentile(da, 2.5)), 4)),
        ("boot_r2_a_hi95", round(float(np.percentile(da, 97.5)), 4)),
        ("boot_r2_b_lo95", round(float(np.percentile(db, 2.5)), 4)),
        ("boot_r2_b_hi95", round(float(np.percentile(db, 97.5)), 4)),
        ("n_required_80pct_power", need if need is not None else "not resolvable"),
        ("n_boot_requested", args.n_boot),
        ("seed", args.seed),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out.to_csv(out_path, index=False)

    width = max(len(str(k)) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{width}}  {v}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
