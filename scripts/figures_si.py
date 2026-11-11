#!/usr/bin/env python3
"""Figures S1-S7, drawn from the result tables and written to figures/.

Uses the same ACS graphics settings as figures_main.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocol_names import protocol_display_name
import cpcmx_arms

from paths import DATA, FIGURES

REPO = DATA
OUT = FIGURES
OUT.mkdir(exist_ok=True)

# Append the five GFN2-xTB/CPCM-X protocols to the metric and score tables.
m = cpcmx_arms.load_metrics(pd.read_csv(REPO / "protocol_metrics.csv").set_index("arm"))
sc = cpcmx_arms.load_scores(pd.read_csv(REPO / "protocol_scores.csv").set_index("System"))
exp = sc["Exp_dG"].astype(float)
st = pd.read_csv(REPO / "structure_statistics.csv")
rm = pd.read_csv(REPO / "rmsd_analysis.csv")

SINGLE, DOUBLE = 3.33, 7.00
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300})
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#c05621", "#276749", "#8a94a6"
REPS = ["r1", "r2", "r3", "r4", "r5"]


def save(fig, n, tight=True):
    fig.savefig(OUT / n, bbox_inches="tight" if tight else None, dpi=300)
    plt.close(fig)


def per_rep(a):
    return [float(x) for x in m.loc[a, "per_rep_r2"].split(" / ")]


# Figure S2: every protocol, ranked
def s2_all_protocols():
    # Two columns keep the labels legible at page size.
    d = m[m.n_sys == 20].sort_values("pool5_r2", ascending=False)
    half = int(np.ceil(len(d) / 2))
    columns = [d.iloc[:half], d.iloc[half:]]

    def fam(a):
        if a.startswith("MMGBSA"): return GREEN
        if a.startswith("GFN2"): return BLUE
        if a.startswith("GFN1"): return "#7aa8d2"
        return ORANGE

    # Sized to the page text width so labels print at the size set here (at least
    # ~6 pt). Compact labels from protocol_names fit in the gutter between columns.
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 8.3))
    fig.subplots_adjust(left=0.32, right=0.995, top=0.955, bottom=0.048, wspace=1.80)
    for ax, chunk, first in zip(axes, columns, (1, half + 1)):
        y = np.arange(len(chunk))
        ax.barh(y, chunk.pool5_r2, color=[fam(a) for a in chunk.index], height=.80)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [protocol_display_name(a, compact=True) for a in chunk.index],
            fontsize=6.0)
        ax.set_ylim(len(chunk) - 0.4, -0.6)
        ax.set_xlim(0, 0.78)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("$R^2$ vs experimental $\\Delta G$", fontsize=7)
        ax.set_title(f"ranks {first}–{first + len(chunk) - 1}", fontsize=6.8,
                     color=GREY, loc="left", pad=3)

    h = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (BLUE, "#7aa8d2", ORANGE, GREEN)]
    axes[1].legend(h, ["GFN2-xTB", "GFN1-xTB", "PM6/PM7", "MM/GBSA"], frameon=False,
                   loc="lower right", fontsize=6.5, handlelength=1.1,
                   handleheight=0.9, labelspacing=0.35)
    fig.suptitle(f"All {len(d)} scoring protocols, ranked by correlation with experiment",
                 fontsize=8.5, y=0.995)
    save(fig, "figureS2.png", tight=False)


# Figure S3: R2 against range across trajectories, all protocols
def s3_accuracy_spread():
    d = m[m.n_sys == 20]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.5, 3.2))

    def fam(a):
        if a.startswith("MMGBSA"): return GREEN, "s"
        if a.startswith("GFN"): return BLUE, "o"
        return ORANGE, "^"
    for a in d.index:
        c, mk = fam(a)
        ax.plot(d.loc[a, "rep_spread"], d.loc[a, "pool5_r2"], mk, color=c,
                ms=3.4, alpha=.8, mew=0)
    for c, mk, lab in ((BLUE, "o", "xTB"), (ORANGE, "^", "PM6/PM7"),
                       (GREEN, "s", "MM/GBSA")):
        ax.plot([], [], mk, color=c, ms=4.5, label=lab)
    ax.set_xlabel("range across trajectories")
    ax.set_ylabel("$R^2$")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "figureS3.png")


# Figure S4: per-trajectory R2
def s4_per_trajectory():
    arms = [a for a in m.index if m.loc[a, "n_sys"] == 20]
    arms = sorted(arms, key=lambda a: -m.loc[a, "pool5_r2"])[:28]
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.0))
    for i, a in enumerate(arms):
        v = per_rep(a)
        ax.plot([min(v), max(v)], [i, i], "-", color=GREY, lw=.9)
        ax.plot(v, [i] * len(v), "o", mfc="white", mec=GREY, mew=.8, ms=3.6)
        ax.plot(m.loc[a, "pool5_r2"], i, "D", color=BLUE, ms=5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(
        [protocol_display_name(a, compact=True) for a in arms], fontsize=6.2)
    ax.set_ylim(len(arms) - 0.4, -0.8)
    ax.set_xlabel("$R^2$ per trajectory (circles) and pooled (diamond)")
    ax.set_title("Twenty-eight best protocols", fontsize=9)
    fig.subplots_adjust(left=0.48, right=0.98, top=0.92, bottom=0.10)
    save(fig, "figureS4.png")


# Figure S1: backbone RMSD of the MD trajectories
def s1_rmsd():
    g = rm.groupby("system").rmsd_mean
    order = g.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(DOUBLE * .82, 3.0))
    for i, s in enumerate(order):
        v = rm[rm.system == s].rmsd_mean.values
        ax.plot([v.min(), v.max()], [i, i], "-", color=GREY, lw=.9)
        ax.plot(v, [i] * len(v), "o", mfc="white", mec=BLUE, mew=.9, ms=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.4)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("backbone RMSD from the first production frame (Å)")
    ax.set_title("Each circle is one of the five trajectories", fontsize=8.5)
    save(fig, "figureS1.png")


# Figure S5: clustering statistics
def s5_clustering():
    tags = ["aa", "bb", "iface_sc", "contacts"]
    names = ["all-atom\nRMSD", "backbone\nRMSD", "interface\nside-chain\nRMSD",
             "interchain\ncontacts"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * .9, 2.9),
                             gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    data = [st[f"basins_{t}"].dropna().values for t in tags]
    bp = ax.boxplot(data, labels=names, widths=.55, patch_artist=True,
                    medianprops=dict(color="black", lw=1.0))
    for p in bp["boxes"]:
        p.set_facecolor(BLUE); p.set_alpha(.55); p.set_edgecolor("none")
    ax.set_ylabel("basins per trajectory")
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=9.5)

    ax = axes[1]
    x = np.arange(len(tags))
    r2v = [m.loc[f"GFN2_GBSA_basin_{t}_unw", "pool5_r2"] for t in tags]
    spv = [m.loc[f"GFN2_GBSA_basin_{t}_unw", "rep_spread"] for t in tags]
    ax.bar(x - .19, r2v, .36, color=BLUE, label="$R^2$")
    ax.bar(x + .19, spv, .36, color=ORANGE, label="range across trajectories")
    for i, (a, b) in enumerate(zip(r2v, spv)):
        ax.text(i - .19, a + .012, f"{a:.3f}", ha="center", fontsize=6.2)
        ax.text(i + .19, b + .012, f"{b:.3f}", ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylim(0, 0.90)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2,
              handletextpad=0.4, columnspacing=0.8)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=9.5, pad=2)
    save(fig, "figureS5.png")


# Figure S6: number of retained waters
def s6_water():
    n = [0, 10, 20, 30, 40]
    v = [m.loc[f"MMGBSA_Nwat{k}", "pool5_r2"] for k in n]
    s = [m.loc[f"MMGBSA_Nwat{k}", "rep_spread"] for k in n]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    ax.plot(n, v, "o-", color=BLUE, ms=5, lw=1.3, label="$R^2$")
    ax.plot(n, s, "s--", color=ORANGE, ms=4.5, lw=1.1,
            label="range across trajectories")
    for a, b in zip(n, v):
        ax.annotate(f"{b:.3f}", (a, b), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.4)
    ax.set_xlabel("explicit interfacial waters retained")
    ax.set_ylabel("$R^2$ / range")
    ax.set_ylim(0, 0.78)
    ax.set_xlim(-4, 44)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33))
    save(fig, "figureS6.png")


# Figure S7: calibration of selected protocols
def s7_calibration_grid():
    arms = [("GFN2-xTB/GBSA, interface basins", "GFN2_GBSA_basin_iface_sc_unw"),
            ("GFN2-xTB/GBSA, all-atom basins", "GFN2_GBSA_basin_aa_unw"),
            ("GFN2-xTB/GBSA, even-10", "GFN2_GBSA_even10"),
            ("GFN2-xTB/ddCOSMO, even-10", "GFN2_COSMO_even10"),
            ("GFN2-xTB/CPCM-X, interface basins", "GFN2_CPCMX_basin_iface_sc_unw"),
            ("GFN2-xTB/CPCM-X, even-10", "GFN2_CPCMX_even10"),
            ("PM6-D3/COSMO, even-10", "PM6_D3_COSMO_even10"),
            ("Nwat-MM/GBSA, 30 waters", "MMGBSA_Nwat30")]
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 4.2))
    letters = "abcdefgh"
    for ax, (lab, a), letter in zip(axes.ravel(), arms, letters):
        d = sc[[a]].join(exp).dropna()
        x, y = d[a].to_numpy(float), d["Exp_dG"].to_numpy(float)
        sl, ic = np.polyfit(x, y, 1)
        ax.scatter(sl * x + ic, y, s=16, color=BLUE, edgecolor="white", lw=.4)
        lo, hi = y.min() - 1.5, y.max() + 1.5
        ax.plot([lo, hi], [lo, hi], ls=(0, (4, 2)), color=GREY, lw=.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{lab}\n$R^2$ = {m.loc[a,'pool5_r2']:.3f}", fontsize=7, pad=2)
        ax.text(0.04, 0.96, letter, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", zorder=4)
        ax.tick_params(labelsize=6.2)
    for ax in axes[1]:
        ax.set_xlabel("calibrated score (kcal mol$^{-1}$)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("experimental $\\Delta G$", fontsize=7)
    fig.tight_layout()
    save(fig, "figureS7.png")


for f in (s1_rmsd, s2_all_protocols, s3_accuracy_spread, s4_per_trajectory,
          s5_clustering, s6_water, s7_calibration_grid):
    f()
    print(" ", f.__name__)
print(f"wrote SI figures to {OUT}")
