#!/usr/bin/env python3
"""Write the interaction energy of every complex, trajectory and protocol as one
long table, data/scores_by_trajectory.csv (protocol, complex, trajectory, score).

The score of a complex in one trajectory is the mean over its evenly spaced
frames or basin centroids (population-weighted for the _wt protocols), in
kcal/mol. Averaging over the five trajectories gives protocol_scores.csv.
"""
from __future__ import annotations

import pandas as pd

import analyze_cpcmx as A
import cpcmx_arms
import harvest_scores as H
from paths import DATA


def cpcmx_matrices():
    exp = A.experimental()
    out = {}
    for s, arm in cpcmx_arms.SETS.items():
        df = A.load_frames(s)
        ok = df[df.status == "ok"]
        out[arm] = A.combine(ok, "cpcmx", "mean")[0].reindex(exp.index)
        if s != "even10":
            out[arm.replace("_unw", "_wt")] = A.combine(ok, "cpcmx", "pop")[0].reindex(exp.index)
    return out


def main():
    mats = H.build_arms()
    mats.update(cpcmx_matrices())
    rows = []
    for protocol, mat in mats.items():
        long = mat.rename_axis("complex").reset_index().melt(
            id_vars="complex", var_name="trajectory", value_name="score")
        long.insert(0, "protocol", protocol)
        rows.append(long.dropna(subset=["score"]))
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["protocol", "complex", "trajectory"])
    out.to_csv(DATA / "scores_by_trajectory.csv", index=False, float_format="%.4f")
    print(f"{out.protocol.nunique()} protocols, {len(out)} rows -> "
          f"{DATA / 'scores_by_trajectory.csv'}")


if __name__ == "__main__":
    main()
