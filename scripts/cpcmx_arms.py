#!/usr/bin/env python3
"""GFN2-xTB/CPCM-X results as five scoring arms in the format of the harvest_scores.py
tables (protocol_metrics.csv, protocol_scores.csv).

Arm keys: GFN2_CPCMX_even10 and GFN2_CPCMX_basin_{aa,bb,iface_sc,contacts}_unw. A
system score is the unweighted mean over frames within a trajectory, then over the
five trajectories. Writes cpcmx_arms_metrics.csv and cpcmx_arms_scores.csv;
load_metrics and load_scores append these rows to the main tables.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_cpcmx as A

from paths import DATA

REPO = DATA
SETS = {"even10": "GFN2_CPCMX_even10", "aa": "GFN2_CPCMX_basin_aa_unw",
        "bb": "GFN2_CPCMX_basin_bb_unw", "iface": "GFN2_CPCMX_basin_iface_sc_unw",
        "contacts": "GFN2_CPCMX_basin_contacts_unw"}


def main():
    exp = A.experimental()
    metrics, scores = [], {}
    for s, arm in SETS.items():
        df = A.load_frames(s)
        mat, _ = A.combine(df[df.status == "ok"], "cpcmx", "mean")
        mat = mat.reindex(exp.index)
        reps = [r for r in A.REPS if r in mat.columns]
        y = exp.to_numpy(float)
        per = [stats.pearsonr(mat[r], y)[0] ** 2 for r in reps]
        pooled = mat[reps].mean(axis=1)
        p3 = [stats.pearsonr(mat[list(c)].mean(axis=1), y)[0] ** 2
              for c in combinations(reps, 3)]
        metrics.append({
            "arm": arm, "n_sys": int(pooled.notna().sum()), "n_rep": len(reps),
            "pool5_r2": stats.pearsonr(pooled, y)[0] ** 2,
            "pool5_rho": stats.spearmanr(pooled, y).correlation,
            "pool5_looq2": np.nan,
            "pool3_mean": float(np.mean(p3)), "pool3_sd": float(np.std(p3)),
            "pool3_min": min(p3), "pool3_max": max(p3),
            "per_rep_r2": " / ".join(f"{v:.3f}" for v in per),
            "rep_spread": max(per) - min(per),
        })
        scores[arm] = pooled
    pd.DataFrame(metrics).to_csv(REPO / "cpcmx_arms_metrics.csv", index=False)
    sc = pd.DataFrame(scores)
    sc.index.name = "System"
    sc.to_csv(REPO / "cpcmx_arms_scores.csv")
    print(pd.DataFrame(metrics)[["arm", "pool5_r2", "rep_spread", "per_rep_r2"]])


def write_weighted_scores() -> pd.DataFrame:
    """Population-weighted basin arms GFN2_CPCMX_basin_*_wt, analogous to the _wt
    arms of the other energy models. Writes cpcmx_arms_scores_wt.csv and returns
    the scores."""
    exp = A.experimental()
    scores = {}
    for s, arm in SETS.items():
        if s == "even10":
            continue
        df = A.load_frames(s)
        mat, _ = A.combine(df[df.status == "ok"], "cpcmx", "pop")
        mat = mat.reindex(exp.index)
        reps = [r for r in A.REPS if r in mat.columns]
        scores[arm.replace("_unw", "_wt")] = mat[reps].mean(axis=1)
    sc = pd.DataFrame(scores)
    sc.index.name = "System"
    sc.to_csv(REPO / "cpcmx_arms_scores_wt.csv")
    return sc


def load_metrics(base: pd.DataFrame) -> pd.DataFrame:
    """protocol_metrics indexed by arm, with the CPCM-X arms appended."""
    extra = pd.read_csv(REPO / "cpcmx_arms_metrics.csv").set_index("arm")
    return pd.concat([base, extra[[c for c in extra.columns if c in base.columns]]])


def load_scores(base: pd.DataFrame) -> pd.DataFrame:
    """protocol_scores indexed by System, with the CPCM-X columns appended."""
    extra = pd.read_csv(REPO / "cpcmx_arms_scores.csv").set_index("System")
    return base.join(extra)


if __name__ == "__main__":
    main()
    write_weighted_scores()
