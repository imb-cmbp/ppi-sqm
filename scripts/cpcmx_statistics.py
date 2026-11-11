#!/usr/bin/env python3
"""Summary statistics for GFN2-xTB/CPCM-X, solvent-model offsets and basin weighting.

Reads the CPCM-X metrics and the per-frame and per-system solvent-offset tables
written by analyze_cpcmx.py and analyze_solvent_offsets.py, the per-frame CPCM-X
JSON results, basin_weights.csv (analyze_basin_weights.py) and the arm
score and metric tables, and writes cpcmx_statistics.csv as (quantity, value).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_cpcmx as A

from paths import DATA

REPO = DATA
SETS = ("iface", "aa", "bb", "contacts", "even10")
REPS = A.REPS
OUT = REPO / "cpcmx_statistics.csv"
rows: list[tuple[str, float]] = []


def put(key, value):
    rows.append((key, float(value)))


def pool5(frames, col):
    """Unweighted pool-5 R2 and spread of a per-frame column."""
    exp = A.experimental()
    mats = {}
    for rep in REPS:
        g = frames[frames.rep == rep].dropna(subset=[col]).groupby("sys")[col].mean()
        mats[rep] = g
    mat = pd.DataFrame(mats).reindex(exp.index)
    y = exp.to_numpy(float)
    per = [stats.pearsonr(mat[r], y)[0] ** 2 for r in REPS]
    p5 = stats.pearsonr(mat.mean(axis=1), y)[0] ** 2
    return p5, max(per) - min(per)


def ols(y, X):
    """Least squares with an intercept: coefficients, their p-values, R2."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X1.T @ X1)
    t = beta / np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return beta, p, r2


def main():
    m = pd.read_csv(REPO / "cpcmx_metrics.csv")
    for _, r in m.iterrows():
        tag = f"{r.set}|{r.method}|{r.weight}"
        put(f"pool5|{tag}", r.pool5_r2)
        put(f"rho|{tag}", r.pool5_rho)
        put(f"spread|{tag}", r.spread)
        put(f"pool3min|{tag}", r.pool3_min)
        put(f"pool3max|{tag}", r.pool3_max)
        put(f"neff|{tag}", r.neff)
    cp = m[(m.method == "GFN2-xTB/CPCM-X") & (m.weight == "mean")].set_index("set")
    gb = m[(m.method == "GFN2-xTB/GBSA") & (m.weight == "mean")].set_index("set")
    gap = (gb.pool5_r2 - cp.pool5_r2)
    put("cpcmx_minus_gbsa_min", gap.min())
    put("cpcmx_minus_gbsa_max", gap.max())
    put("cpcmx_spread_min", cp.spread.min())
    put("cpcmx_spread_max", cp.spread.max())
    put("gbsa_spread_min", gb.spread.min())
    put("gbsa_spread_max", gb.spread.max())
    best = m[m.method == "GFN2-xTB/CPCM-X"].sort_values("pool5_r2").iloc[-1]
    put("cpcmx_best_any_weight", best.pool5_r2)

    fr = pd.read_csv(REPO / "solvent_offsets_frames.csv")
    fr["cond"] = fr.energy_conductor
    fr["post"] = fr.cpcmx - fr.energy_conductor
    for s in SETS:
        d = fr[fr.set == s]
        p5, sp = pool5(d, "cond")
        put(f"conductor_only_pool5|{s}", p5)
        p5, sp = pool5(d, "post")
        put(f"postscf_only_pool5|{s}", p5)
        off = d.cpcmx - d.gfn2_gbsa
        dev = (off - off.groupby(d.sys).transform("median")).abs()
        put(f"frame_offset_median|{s}", off.median())
        put(f"frame_offset_maxdev|{s}", dev.max())
    put("frames_total", len(fr))

    # QC from the per-frame JSON: species with repeated conductor SCFs and
    # whether the first two energies agree.
    reruns = agree = unstable = 0
    for jf in A.V2.rglob("gfn2_cpcmx.json"):
        d = json.loads(jf.read_text())
        for n in ("a", "b", "ab"):
            k = d.get(f"n_scf_runs_{n}")
            if k and k > 1:
                reruns += 1
                e = d.get(f"scf_energies_{n}_Eh") or []
                agree += int(len(e) >= 2 and abs(e[0] - e[1]) < 1e-4)
    put("qc_species_repeated", reruns)
    put("qc_species_repeated_agree_first", agree)
    put("qc_frames_ok", int((fr.status == "ok").sum()))

    # System level: averaged over frames and replicas within a set, then over sets.
    sy = pd.read_csv(REPO / "solvent_offsets_systems.csv").set_index("sys")
    put("sys_r2_cpcmx", stats.pearsonr(sy.cpcmx, sy.exp)[0] ** 2)
    put("sys_r2_gbsa", stats.pearsonr(sy.gfn2_gbsa, sy.exp)[0] ** 2)
    put("off_cpcmx_mean", sy.off_cpcmx.mean())
    put("off_cpcmx_sd", sy.off_cpcmx.std())
    put("off_cpcmx_min", sy.off_cpcmx.min())
    put("off_cpcmx_max", sy.off_cpcmx.max())
    put("off_cosmo_mean", sy.off_cosmo.mean())
    put("off_cosmo_sd", sy.off_cosmo.std())
    put("off_cosmo_min", sy.off_cosmo.min())
    put("off_cosmo_max", sy.off_cosmo.max())

    sb = sy.n_saltbridge_x.to_numpy(float)
    for tag, y in (("cpcmx", sy.off_cpcmx), ("cosmo", sy.off_cosmo),
                   ("cond", sy.off_cond), ("post", sy.post_scf)):
        y = y.to_numpy(float)
        r, p = stats.pearsonr(sb, y)
        put(f"sb_r|{tag}", r)
        put(f"sb_p|{tag}", p)
        put(f"sb_rho|{tag}", stats.spearmanr(sb, y).correlation)
        slope = stats.linregress(sb, y).slope
        put(f"sb_slope|{tag}", slope)
        put(f"sb_r2|{tag}", r * r)

    # CPCM-X offset regressed on salt bridges plus one other descriptor at a time.
    for cov in ("q_complex", "outside_cpcmx_training", "n_iface_charged", "n_atoms",
                "n_charged_res", "charge_density"):
        beta, p, r2 = ols(sy.off_cpcmx.to_numpy(float),
                          np.column_stack([sb, sy[cov].to_numpy(float)]))
        put(f"partial_sb_slope|{cov}", beta[1])
        put(f"partial_sb_p|{cov}", p[1])
        put(f"partial_cov_p|{cov}", p[2])
    beta, p, r2 = ols(sy.off_cosmo.to_numpy(float),
                      np.column_stack([sb, sy.n_atoms.to_numpy(float)]))
    put("cosmo_sb_atoms_r2", r2)
    put("cosmo_sb_atoms_sb_slope", beta[1])
    put("cosmo_sb_atoms_atom_slope", beta[2])
    put("cosmo_sb_atoms_sb_p", p[1])

    # Family-wise permutation test over every descriptor that varies.
    feats = [c for c in ("q_complex", "q_abs_sum", "q_sq_sum", "q_product", "q_complex_sq",
                         "q_max_abs", "outside_cpcmx_training", "n_atoms", "n_res",
                         "n_charged_res", "charged_frac", "n_iface_charged",
                         "iface_net_charge_abs", "n_saltbridge_x", "n_samesign_x",
                         "n_hbond_x", "n_contacts_x", "charge_density", "near_degenerate")
             if sy[c].std() > 0]
    put("n_descriptors", len(feats))
    X = sy[feats].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    y = sy.off_cpcmx.to_numpy(float)

    def maxabs(yv):
        yz = (yv - yv.mean()) / yv.std()
        return np.max(np.abs(X.T @ yz / len(yz)))

    obs_sb = abs(stats.pearsonr(sb, y)[0])
    rng = np.random.default_rng(20260924)
    perm = np.array([maxabs(rng.permutation(y)) for _ in range(20000)])
    single = np.array([abs(stats.pearsonr(sb, rng.permutation(y))[0]) for _ in range(20000)])
    put("perm_p_single", (1 + (single >= obs_sb).sum()) / (1 + len(single)))
    put("perm_p_familywise", (1 + (perm >= obs_sb).sum()) / (1 + len(perm)))

    # Correlations of descriptors and offsets with experiment and with the GBSA score.
    put("r_sb_exp", stats.pearsonr(sb, sy.exp)[0])
    put("p_sb_exp", stats.pearsonr(sb, sy.exp)[1])
    put("r_hb_exp", stats.pearsonr(sy.n_hbond_x, sy.exp)[0])
    put("p_hb_exp", stats.pearsonr(sy.n_hbond_x, sy.exp)[1])
    put("r_hb_gbsa", stats.pearsonr(sy.n_hbond_x, sy.gfn2_gbsa)[0])
    put("r_hb_cpcmx", stats.pearsonr(sy.n_hbond_x, sy.cpcmx)[0])
    put("r_off_exp", stats.pearsonr(sy.off_cpcmx, sy.exp)[0])
    put("r_off_gbsa", stats.pearsonr(sy.off_cpcmx, sy.gfn2_gbsa)[0])
    put("r_outside_off", stats.pearsonr(sy.outside_cpcmx_training, sy.off_cpcmx)[0])
    put("p_outside_off", stats.pearsonr(sy.outside_cpcmx_training, sy.off_cpcmx)[1])
    put("r_density_extra", stats.pearsonr(sy.charge_density, sy.extra_err_cpcmx)[0])
    put("p_density_extra", stats.pearsonr(sy.charge_density, sy.extra_err_cpcmx)[1])
    put("r_gbsa_alpb_sys", stats.pearsonr(sy.gfn2_gbsa, sy.gfn2_alpb)[0])
    put("r_gbsa_cpcmx_sys", stats.pearsonr(sy.gfn2_gbsa, sy.cpcmx)[0])
    put("r_gbsa_cosmo_sys", stats.pearsonr(sy.gfn2_gbsa, sy.gfn2_cosmo)[0])

    # Leave one complex out: change in the CPCM-X minus GBSA R2 gap.
    base = put_gap = (stats.pearsonr(sy.cpcmx, sy.exp)[0] ** 2
                      - stats.pearsonr(sy.gfn2_gbsa, sy.exp)[0] ** 2)
    put("sys_gap", base)
    for s in ("1EMV", "1AY7", "2OOB"):
        k = sy.drop(index=s)
        put(f"loo_gap|{s}", stats.pearsonr(k.cpcmx, k.exp)[0] ** 2
            - stats.pearsonr(k.gfn2_gbsa, k.exp)[0] ** 2)

    # Salt-bridge dependence within each sampling set.
    for s in SETS:
        g = fr[fr.set == s].groupby("sys")[["off_cpcmx", "off_cosmo", "n_saltbridge_x"]].mean()
        put(f"sb_r_set|cpcmx|{s}", stats.pearsonr(g.n_saltbridge_x, g.off_cpcmx)[0])
        put(f"sb_r_set|cosmo|{s}", stats.pearsonr(g.n_saltbridge_x, g.off_cosmo)[0])

    # Frame-level correlation within complexes, set and system means removed.
    g = fr.groupby(["set", "sys"])
    x = fr.n_saltbridge_x - g.n_saltbridge_x.transform("mean")
    put("within_sb_r|cpcmx", np.corrcoef(x, fr.off_cpcmx - g.off_cpcmx.transform("mean"))[0, 1])
    put("within_sb_r|cosmo", np.corrcoef(x, fr.off_cosmo - g.off_cosmo.transform("mean"))[0, 1])

    for s in ("1AY7", "1EMV", "2OOB", "2HLE"):
        put(f"sb_count|{s}", sy.loc[s, "n_saltbridge_x"])
        put(f"extra_err|{s}", sy.loc[s, "extra_err_cpcmx"])
    put("sb_max", sy.n_saltbridge_x.max())
    put("sb_min", sy.n_saltbridge_x.min())

    # Wall time of one CPCM-X interaction energy (three conductor SCFs and three cpx runs).
    secs = []
    for jf in A.V2.rglob("gfn2_cpcmx.json"):
        d = json.loads(jf.read_text())
        if d.get("status") == "ok" and d.get("seconds"):
            secs.append(float(d["seconds"]))
    put("seconds_median_4threads", np.median(secs))

    # Basin weighting over every basin protocol (analyze_basin_weights.py).
    bw = pd.read_csv(REPO / "basin_weights.csv")
    r2w = bw.pivot_table(index=["scheme", "method"], columns="weight", values="r2")
    spw = bw.pivot_table(index=["scheme", "method"], columns="weight", values="spread")
    nfw = bw.pivot_table(index=["scheme", "method"], columns="weight", values="neff")
    d_b = r2w.boltz - r2w["mean"]
    d_bs = spw.boltz - spw["mean"]
    d_p = r2w["pop"] - r2w["mean"]
    put("bw_n_protocols", len(r2w))
    put("bw_boltz_up", int((d_b > 0).sum()))
    put("bw_boltz_down", int((d_b < 0).sum()))
    put("bw_boltz_median_dr2", d_b.median())
    put("bw_boltz_median_dspread", d_bs.median())
    put("bw_boltz_wider", int((d_bs > 0).sum()))
    put("bw_boltz_neff_max", nfw.boltz.max())
    put("bw_pop_max_abs_dr2", d_p.abs().max())
    put("bw_pop_median_dr2", d_p.median())
    fam = pd.Series(["GFN2" if m.startswith("gfn2") else "GFN1" if m.startswith("gfn1")
                     else "MM" if m == "mmgbsa" else "PM7" if m.startswith("pm7") else "PM6"
                     for _, m in r2w.index], index=r2w.index)
    for f, g in d_b.groupby(fam):
        put(f"bw_boltz_median_dr2|{f}", g.median())
    for key in (("iface_sc", "gfn2_gbsa"), ("iface_sc", "pm7_cosmo2"),
                ("iface_sc", "gfn1_gbsa"), ("iface_sc", "mmgbsa")):
        put(f"bw_mean|{key[0]}|{key[1]}", r2w.loc[key, "mean"])
        put(f"bw_boltz|{key[0]}|{key[1]}", r2w.loc[key, "boltz"])
        put(f"bw_spread_mean|{key[0]}|{key[1]}", spw.loc[key, "mean"])
        put(f"bw_spread_boltz|{key[0]}|{key[1]}", spw.loc[key, "boltz"])
    gfn2 = fam == "GFN2"
    put("bw_pop_max_abs_dr2|GFN2", d_p[gfn2].abs().max())

    # GFN2 minus GFN1 at even sampling, and CPCM-X residuals alongside six main protocols.
    pm = pd.read_csv(REPO / "protocol_metrics.csv").set_index("arm")["pool5_r2"]
    gb = [pm[f"GFN2_{s}_even10"] - pm[f"GFN1_{s}_even10"] for s in ("GBSA", "ALPB")]
    put("gfn2_minus_gfn1_gbsa_alpb_min", min(gb))
    put("gfn2_minus_gfn1_gbsa_alpb_max", max(gb))
    put("gfn2_minus_gfn1_cosmo", pm["GFN2_COSMO_even10"] - pm["GFN1_COSMO_even10"])
    arms = pd.read_csv(REPO / "cpcmx_arms_scores.csv").set_index("System")
    main = pd.read_csv(REPO / "protocol_scores.csv").set_index("System")
    exp = main.Exp_dG.astype(float)

    def resid(x):
        slope, icpt = np.polyfit(x.to_numpy(float), exp.loc[x.index].to_numpy(float), 1)
        r = exp.loc[x.index] - (slope * x + icpt)
        return r, r / r.std(ddof=1)

    r, z = resid(arms["GFN2_CPCMX_basin_iface_sc_unw"])
    put("cpcmx_resid|1EMV", r["1EMV"])
    put("cpcmx_z|1EMV", z["1EMV"])
    put("cpcmx_resid|2HLE", r["2HLE"])
    six = ["MMGBSA_Nwat30", "PM6_D3H4_COSMO_NOFC_even10", "GFN2_ALPB_even10",
           "GFN2_GBSA_even10", "GFN2_GBSA_basin_aa_unw", "GFN2_GBSA_basin_iface_sc_unw"]
    hle = [resid(main[a])[0]["2HLE"] for a in six]
    put("six_resid_2HLE_min", min(hle))
    put("six_resid_2HLE_max", max(hle))
    em = [resid(main[a])[0]["1EMV"] for a in six]
    put("six_resid_1EMV_min", min(em))
    put("six_resid_1EMV_max", max(em))
    cm = pd.read_csv(REPO / "cpcmx_arms_metrics.csv").set_index("arm")
    put("cpcmx_even10_rho", cm.loc["GFN2_CPCMX_even10", "pool5_rho"])
    put("cpcmx_aa_gain", cm.loc["GFN2_CPCMX_basin_aa_unw", "pool5_r2"]
        - cm.loc["GFN2_CPCMX_even10", "pool5_r2"])

    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(OUT, index=False)
    for k, v in rows:
        if not k.startswith(("pool5|", "rho|", "spread|", "pool3", "neff|")):
            print(f"{k:45s} {v:10.4f}")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
