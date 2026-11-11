#!/usr/bin/env python3
"""Interface descriptors versus solvent-model offsets relative to GFN2-xTB/GBSA.

For every frame scored with CPCM-X, computes descriptors of the truncated
interface model (partner charges, size, explicit waters, cross-interface salt
bridges and hydrogen bonds, interfacial charged residues) and correlates them,
between and within systems, with E(solvent) - E(GBSA) for CPCM-X and ddCOSMO
on the same frame; only offsets that vary between complexes change the ranking.
Writes solvent_offsets_{frames,systems,descriptor_corr}.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import analyze_cpcmx as A

from paths import DATA

REPO = DATA
SHELLS = A.SCR
SHELL_ROOT = {
    "iface": SHELLS / "_qmmm_trunc_propka_clust_iface_sc_gt5",
    "aa": SHELLS / "_qmmm_trunc_propka_clust_aa_gt5",
    "bb": SHELLS / "_qmmm_trunc_propka_clust_bb_gt5",
    "contacts": SHELLS / "_qmmm_trunc_propka_clust_contacts_gt5",
    "even10": SHELLS / "_qmmm_trunc_propka",
}
ANION = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
CATION = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
          ("HIP", "ND1"), ("HIP", "NE2")}
CHARGED_RES = {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIP": 1}
WATER = {"HOH", "WAT"}
SB_CUT, HB_CUT, CONTACT_CUT, IFACE_CUT = 4.0, 3.5, 4.0, 5.0


def parse_shell(pdb):
    rows = []
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rows.append((line[21], line[17:20].strip(), int(line[22:26]),
                     line[12:16].strip(), line[76:78].strip() or line[12:16].strip()[0],
                     float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return pd.DataFrame(rows, columns=["chain", "res", "resid", "name", "el", "x", "y", "z"])


def descriptors(pdb, meta):
    d = parse_shell(pdb)
    xyz = d[["x", "y", "z"]].to_numpy()
    prot = ~d.res.isin(WATER)
    heavy = d.el != "H"
    A_ = (d.chain == "A").to_numpy()
    B_ = ~A_

    def cross_pairs(mask_i, mask_j, cut):
        ii, jj = np.where(mask_i)[0], np.where(mask_j)[0]
        if len(ii) == 0 or len(jj) == 0:
            return []
        t = cKDTree(xyz[jj])
        out = []
        for k, nbrs in zip(ii, t.query_ball_point(xyz[ii], cut)):
            out.extend((k, jj[n]) for n in nbrs)
        return out

    key = list(zip(d.res, d.name))
    anion = np.array([k in ANION for k in key])
    cation = np.array([k in CATION for k in key])
    polar = d.el.isin(["N", "O"]).to_numpy() & prot.to_numpy()

    sb = set()
    for i, j in cross_pairs(anion & A_, cation & B_, SB_CUT) + cross_pairs(cation & A_, anion & B_, SB_CUT):
        sb.add((d.chain[i], d.resid[i], d.chain[j], d.resid[j]))
    same_sign = set()
    for i, j in cross_pairs(anion & A_, anion & B_, 6.0) + cross_pairs(cation & A_, cation & B_, 6.0):
        same_sign.add((d.resid[i], d.resid[j]))
    hb = len(cross_pairs(polar & A_, polar & B_, HB_CUT))
    contacts = len(cross_pairs(heavy.to_numpy() & prot.to_numpy() & A_,
                               heavy.to_numpy() & prot.to_numpy() & B_, CONTACT_CUT))

    # charged residues with any heavy atom within IFACE_CUT of the other partner
    near = set()
    for i, j in cross_pairs(heavy.to_numpy() & prot.to_numpy() & A_,
                            heavy.to_numpy() & prot.to_numpy() & B_, IFACE_CUT):
        near.add((d.chain[i], d.resid[i]))
        near.add((d.chain[j], d.resid[j]))
    res_tab = d[prot].drop_duplicates(["chain", "resid"])[["chain", "resid", "res"]]
    res_tab["q"] = res_tab.res.map(CHARGED_RES).fillna(0)
    res_tab["iface"] = [(c, r) in near for c, r in zip(res_tab.chain, res_tab.resid)]
    iq = res_tab[res_tab.iface]
    qa, qb = int(meta["charge_a"]), int(meta["charge_b"])
    return {
        "q_a": qa, "q_b": qb, "q_complex": qa + qb,
        "q_abs_sum": abs(qa) + abs(qb), "q_sq_sum": qa * qa + qb * qb,
        "q_product": qa * qb, "q_complex_sq": (qa + qb) ** 2,
        "q_max_abs": max(abs(qa), abs(qb)),
        "outside_cpcmx_training": int(max(abs(qa), abs(qb)) > 2),
        "n_atoms": int(meta["n_atoms"]), "n_waters": int(meta.get("n_waters", 0)),
        "n_res": int(meta.get("n_res_a", 0)) + int(meta.get("n_res_b", 0)),
        "n_charged_res": int((res_tab.q != 0).sum()),
        "charged_frac": float((res_tab.q != 0).mean()),
        "n_iface_charged": int((iq.q != 0).sum()),
        "iface_net_charge_abs": float(abs(iq.q.sum())),
        "n_saltbridge_x": len(sb), "n_samesign_x": len(same_sign),
        "n_hbond_x": hb, "n_contacts_x": contacts,
        "charge_density": (abs(qa) + abs(qb)) / int(meta["n_atoms"]),
    }


def load(set_name):
    df = A.load_frames(set_name)
    extra = []
    for rep, s, f in zip(df.rep, df.sys, df.frame):
        jf = A.V2 / set_name / rep / s / f"f{f}" / "gfn2_cpcmx.json"
        j = json.loads(jf.read_text())
        extra.append({k: j.get(k) for k in ("energy_conductor", "dG_res", "dG_cc",
                                            "gap_cosmo_a_eV", "gap_cosmo_b_eV")})
    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    feats = []
    for rep, s, f in zip(df.rep, df.sys, df.frame):
        sh = SHELL_ROOT[set_name] / rep / s / "c8" / "wet" / f"f{f}"
        meta = json.loads((sh / "capped.trunc.json").read_text())
        feats.append(descriptors(sh / "capped.pdb", meta))
    df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    df["set"] = set_name
    df["post_scf"] = df.cpcmx - df.energy_conductor
    df["off_cpcmx"] = df.cpcmx - df.gfn2_gbsa
    df["off_cond"] = df.energy_conductor - df.gfn2_gbsa
    df["off_cosmo"] = df.gfn2_cosmo - df.gfn2_gbsa
    df["near_degenerate"] = ((df.gap_cosmo_a_eV < 0.5) | (df.gap_cosmo_b_eV < 0.5)).astype(int)
    return df


FEATURES = ["q_complex", "q_abs_sum", "q_sq_sum", "q_product", "q_complex_sq",
            "q_max_abs", "outside_cpcmx_training", "n_atoms", "n_waters", "n_res",
            "n_charged_res", "charged_frac", "n_iface_charged", "iface_net_charge_abs",
            "n_saltbridge_x", "n_samesign_x", "n_hbond_x", "n_contacts_x",
            "charge_density", "near_degenerate"]


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0:
        return np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x[ok], y[ok])
    rho = stats.spearmanr(x[ok], y[ok]).correlation
    return r, p, rho


def main():
    exp = A.experimental()
    frames = pd.concat([load(s) for s in ("iface", "aa", "bb", "contacts", "even10")],
                       ignore_index=True)
    frames.to_csv(REPO / "solvent_offsets_frames.csv", index=False)
    print(f"frames {len(frames)}")

    # Per system: pool every frame of every replica within a sampling set,
    # then average the set-level system means over the five sets.
    per_set = (frames.groupby(["set", "sys"])
               [["cpcmx", "gfn2_gbsa", "gfn2_alpb", "gfn2_cosmo", "energy_conductor",
                 "post_scf", "dG_res", "cds", "off_cpcmx", "off_cond", "off_cosmo", "q_a", "q_b"]
                + FEATURES]
               .mean())
    sysm = per_set.groupby(level="sys").mean()
    sysm["exp"] = exp.reindex(sysm.index)
    # Residual of each method against experiment after its own linear fit.
    for m in ("cpcmx", "gfn2_gbsa", "gfn2_cosmo"):
        slope, icpt, *_ = stats.linregress(sysm[m], sysm.exp)
        sysm[f"res_{m}"] = sysm.exp - (icpt + slope * sysm[m])
    sysm["extra_err_cpcmx"] = sysm.res_cpcmx.abs() - sysm.res_gfn2_gbsa.abs()
    sysm["extra_err_cosmo"] = sysm.res_gfn2_cosmo.abs() - sysm.res_gfn2_gbsa.abs()

    # Leave-one-system-out change in the CPCM-X minus GBSA R^2 gap.
    def r2(a, b):
        return stats.pearsonr(a, b)[0] ** 2
    base = r2(sysm.cpcmx, sysm.exp) - r2(sysm.gfn2_gbsa, sysm.exp)
    loo = {}
    for s in sysm.index:
        k = sysm.drop(index=s)
        loo[s] = (r2(k.cpcmx, k.exp) - r2(k.gfn2_gbsa, k.exp)) - base
    sysm["loo_gap_change"] = pd.Series(loo)
    sysm.to_csv(REPO / "solvent_offsets_systems.csv")

    print(f"\nsystem-level R2 (mean over 5 sets): CPCM-X {r2(sysm.cpcmx, sysm.exp):.3f}  "
          f"GBSA {r2(sysm.gfn2_gbsa, sysm.exp):.3f}  ddCOSMO {r2(sysm.gfn2_cosmo, sysm.exp):.3f}")
    print(f"offset CPCM-X-GBSA by system: mean {sysm.off_cpcmx.mean():.1f}  sd {sysm.off_cpcmx.std():.1f}  "
          f"range {sysm.off_cpcmx.min():.1f}..{sysm.off_cpcmx.max():.1f}")
    print(f"  conductor part sd {sysm.off_cond.std():.1f}, post-SCF part sd {sysm.post_scf.std():.1f}, "
          f"r(conductor part, post-SCF) {sysm.off_cond.corr(sysm.post_scf):.2f}")
    print(f"offset ddCOSMO-GBSA by system: mean {sysm.off_cosmo.mean():.1f}  sd {sysm.off_cosmo.std():.1f}")
    print(f"r(offset CPCM-X, exp) {sysm.off_cpcmx.corr(sysm.exp):.2f}   "
          f"r(offset ddCOSMO, exp) {sysm.off_cosmo.corr(sysm.exp):.2f}   "
          f"r(GBSA, exp) {sysm.gfn2_gbsa.corr(sysm.exp):.2f}")
    print(f"r(offset CPCM-X, GBSA score) {sysm.off_cpcmx.corr(sysm.gfn2_gbsa):.2f}   "
          f"r(offset ddCOSMO, GBSA score) {sysm.off_cosmo.corr(sysm.gfn2_gbsa):.2f}")

    print("\nsystem-level correlation of features with the offset (n=20)")
    print(f"{'feature':24s} {'CPCM-X r':>9s} {'p':>7s} {'rho':>6s} | {'ddCOSMO r':>9s} {'p':>7s} {'rho':>6s} | "
          f"{'CPCM-X extra err r':>18s} {'p':>7s}")
    rows = []
    for f in FEATURES:
        r1, p1, s1 = corr(sysm[f].to_numpy(float), sysm.off_cpcmx.to_numpy(float))
        r2_, p2, s2 = corr(sysm[f].to_numpy(float), sysm.off_cosmo.to_numpy(float))
        r3, p3, _ = corr(sysm[f].to_numpy(float), sysm.extra_err_cpcmx.to_numpy(float))
        rows.append((f, r1, p1, s1, r2_, p2, s2, r3, p3))
        print(f"{f:24s} {r1:9.2f} {p1:7.3f} {s1:6.2f} | {r2_:9.2f} {p2:7.3f} {s2:6.2f} | {r3:18.2f} {p3:7.3f}")
    pd.DataFrame(rows, columns=["feature", "r_off_cpcmx", "p_off_cpcmx", "rho_off_cpcmx",
                                "r_off_cosmo", "p_off_cosmo", "rho_off_cosmo",
                                "r_extra_err_cpcmx", "p_extra_err_cpcmx"]).to_csv(
        REPO / "solvent_offsets_descriptor_corr.csv", index=False)

    # Within-system: frame offsets and frame features, each minus its system-set
    # mean, which removes between-complex confounding.
    print("\nwithin-system (frame-level, system-and-set means removed), pooled over sets")
    w = frames.copy()
    g = w.groupby(["set", "sys"])
    print(f"{'feature':24s} {'CPCM-X r':>9s} {'ddCOSMO r':>10s}  n={len(w)}")
    for f in ["n_saltbridge_x", "n_samesign_x", "n_hbond_x", "n_contacts_x", "n_iface_charged",
              "iface_net_charge_abs", "n_waters", "n_atoms"]:
        x = (w[f] - g[f].transform("mean")).to_numpy(float)
        y1 = (w.off_cpcmx - g.off_cpcmx.transform("mean")).to_numpy(float)
        y2 = (w.off_cosmo - g.off_cosmo.transform("mean")).to_numpy(float)
        print(f"{f:24s} {corr(x, y1)[0]:9.2f} {corr(x, y2)[0]:10.2f}")

    print("\nsystems ranked by extra error of CPCM-X over GBSA (|residual| difference, kcal/mol)")
    cols = ["exp", "q_a", "q_b", "q_sq_sum", "n_saltbridge_x", "n_iface_charged", "near_degenerate",
            "off_cpcmx", "off_cosmo", "res_gfn2_gbsa", "res_cpcmx", "extra_err_cpcmx", "loo_gap_change"]
    print(sysm.sort_values("extra_err_cpcmx", ascending=False)[cols].round(2).to_string())


if __name__ == "__main__":
    main()
