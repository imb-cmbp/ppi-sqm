#!/usr/bin/env python3
"""Main figures, drawn from protocol_metrics.csv, protocol_scores.csv and related tables.

PNGs are written to figures/ (figure1.png to figure8.png). Graphics follow the ACS/JCIM specification:
  - Arial/Helvetica lettering, at least 7 pt at final size
  - line weights of at least 0.5 pt; 300 dpi colour
  - single column 3.33 in, double column 4.17-7.00 in
  - values printed beside bars and markers; legends inside the artwork
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

SINGLE, DOUBLE = 3.33, 7.00
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#c05621", "#276749", "#8a94a6"
TEAL = "#2c7a7b"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def per_rep(arm):
    return [float(x) for x in m.loc[arm, "per_rep_r2"].split(" / ")]


def r2(a):
    return float(m.loc[a, "pool5_r2"])


def sp(a):
    return float(m.loc[a, "rep_spread"])


# Figure 1
def _trim(img):
    """Crop the empty margins around a PyMOL rendering."""
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0.02
    else:
        mask = np.any(img[:, :, :3] < 0.99, axis=2)
    if not mask.any():
        return img
    rows, cols = np.where(mask)
    pad = 6
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0] - 1)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1] - 1)
    return img[r0:r1 + 1, c0:c1 + 1]


def figure1_overview():
    """Example interfaces and the truncated model (a-d), the amide caps (e), and
    pooled R2 of PM6/COSMO with each empirical correction (f)."""
    fig = plt.figure(figsize=(DOUBLE, 4.45))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.82, 1.0],
                          hspace=0.16, wspace=0.10)

    mol = [("panelD_1EMV.png", "a", "1EMV, tightest binder"),
           ("panelD_2OOB.png", "b", "2OOB, weakest binder"),
           ("panelA_complex.png", "c", "1ACB, interface selected"),
           ("panelB_truncated.png", "d", "1ACB, capped model scored")]
    for i, (fn, lab, sub) in enumerate(mol):
        ax = fig.add_subplot(gs[0, i])
        p = OUT / "panels" / fn
        if p.exists():
            ax.imshow(_trim(mpimg.imread(p)))
        ax.axis("off")
        ax.text(0.0, 1.06, lab, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="top")
        ax.text(0.5, -0.03, sub, transform=ax.transAxes, fontsize=6.6,
                ha="center", va="top", color="#222222")

    ax = fig.add_subplot(gs[1, :2])
    p = OUT / "panels" / "panelC_caps.png"
    if p.exists():
        ax.imshow(_trim(mpimg.imread(p)))
    ax.axis("off")
    ax.text(-0.06, 1.04, "e", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top")
    ax.text(0.5, -0.02, "one retained residue and its two amide caps (yellow)",
            transform=ax.transAxes, fontsize=6.6, ha="center", va="top",
            color="#222222")

    ax = fig.add_subplot(gs[1, 2:])
    pos = ax.get_position()                 # clear gutter from the panel-e molecule
    ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    items = [("none", "PM6_COSMO_even10"), ("H4", "PM6_H4_COSMO_CUBY_even10"),
             ("D3+H4", "PM6_D3H4_COSMO_NOFC_even10"), ("D3", "PM6_D3_COSMO_even10")]
    items = [(l, a) for l, a in items if a in m.index]
    vals = [r2(a) for _, a in items]
    bars = ax.bar(range(len(items)), vals, width=.66,
                  color=[GREY, GREY, BLUE, BLUE], edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([l for l, _ in items], fontsize=7.5)
    ax.set_xlabel("empirical correction added to PM6/COSMO", fontsize=7.8)
    ax.set_ylabel("$R^2$ vs experimental $\\Delta G$", fontsize=7.8)
    ax.set_ylim(0, 0.58)
    ax.tick_params(labelsize=7)
    ax.text(-0.24, 1.04, "f", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top")
    save(fig, "figure1.png")


# Figure 2
def figure2_even10():
    """Pooled R2 of every energy model with even-10 sampling, with the MM/GBSA reference."""
    e = m[[i.endswith("_even10") for i in m.index]].sort_values("pool5_r2")
    # Full (non-compact) protocol names.
    lab = [protocol_display_name(a, compact=False, omit_even10=True)
           for a in e.index]
    vals = list(e.pool5_r2)

    def fam(a):
        if a.startswith("GFN2"): return 0
        if a.startswith("GFN1"): return 1
        if a.startswith("PM7"): return 3
        return 2
    cols = [BLUE, "#63a4d8", ORANGE, "#e39a62"]
    f = [fam(a) for a in e.index]

    # Canvas sized so that labels stay close to 7 pt at final size.
    fig, ax = plt.subplots(figsize=(6.5, 5.9))
    y = np.arange(len(e))
    ax.barh(y, vals, color=[cols[i] for i in f], edgecolor="white",
            linewidth=0.5, height=.74)
    # MM/GBSA reference as a vertical line, labelled in the legend
    ref = r2("MMGBSA_Nwat30")
    for i, v in enumerate(vals):
        x = v + .006
        if x < ref < x + .07:               # value label would sit on the rule
            x = ref + .008
        ax.text(x, i, f"{v:.3f}", va="center", fontsize=6.5)
    ax.axvline(ref, color=GREEN, lw=1.3, ls=(0, (4, 2)), zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7.0)
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("pooled $R^2$ vs experimental $\\Delta G$")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cols[i],
                             ec="white", lw=.5) for i in range(4)]
    handles.append(Line2D([0], [0], color=GREEN, lw=1.3, ls=(0, (4, 2))))
    # Legend placed left of the reference line so that the line does not cross it.
    ax.legend(handles,
              ["GFN2-xTB", "GFN1-xTB", "PM6", "PM7",
               f"Nwat-MM/GBSA ({ref:.3f})"],
              frameon=False, loc="lower right", bbox_to_anchor=(0.72, 0.02),
              borderaxespad=0, handlelength=1.6)
    fig.subplots_adjust(left=0.46, right=0.98, top=0.98, bottom=0.08)
    save(fig, "figure2.png")


# Figure 3: range of R2 across trajectories
def figure3_spread():
    arms = [("GFN2-xTB/GBSA, interface basins", "GFN2_GBSA_basin_iface_sc_unw"),
            ("GFN2-xTB/GBSA, even-10", "GFN2_GBSA_even10"),
            ("GFN2-xTB/ALPB, even-10", "GFN2_ALPB_even10"),
            ("GFN2-xTB/CPCM-X, even-10", "GFN2_CPCMX_even10"),
            ("GFN2-xTB/ddCOSMO, even-10", "GFN2_COSMO_even10"),
            ("Nwat-MM/GBSA, 30 waters", "MMGBSA_Nwat30"),
            ("PM6-D3H4/COSMO, even-10", "PM6_D3H4_COSMO_NOFC_even10"),
            ("GFN1-xTB/ALPB, even-10", "GFN1_ALPB_even10")]
    arms = [(l, a) for l, a in arms if a in m.index]
    fig, ax = plt.subplots(figsize=(DOUBLE * .78, 2.9))
    for i, (lab, a) in enumerate(arms):
        v = per_rep(a)
        ax.plot([min(v), max(v)], [i, i], "-", color=GREY, lw=1.0, zorder=1)
        for end in (min(v), max(v)):          # end caps delimit the range
            ax.plot([end, end], [i - 0.16, i + 0.16], "-", color=GREY, lw=1.0,
                    zorder=1)
        ax.plot(r2(a), i, "D", color=BLUE, ms=6, zorder=3)
        ax.text(max(max(v), r2(a)) + .022, i, f"range {sp(a):.3f}", va="center",
                fontsize=7, color="#333333")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([l for l, _ in arms], fontsize=7.5)
    ax.set_ylim(len(arms) - 0.45, -0.65)
    ax.set_xlim(0.28, 0.96)
    ax.set_xlabel("$R^2$ vs experimental $\\Delta G$")
    h_range = Line2D([0], [0], color=GREY, lw=1.2)
    h_pool = Line2D([0], [0], marker="D", color=BLUE, ls="none", ms=6)
    ax.legend([h_range, h_pool],
              ["range of the five trajectories", "pooled over five trajectories"],
              frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
              columnspacing=1.2, fontsize=7.5, borderaxespad=0.2)
    save(fig, "figure3.png")


# Figure 4: basin centroids against even-10 sampling
def figure4_clustering_gain():
    pairs = [("GFN2-xTB/GBSA", "GFN2_GBSA"), ("GFN2-xTB/ALPB", "GFN2_ALPB"),
             ("GFN2-xTB/CPCM-X", "GFN2_CPCMX"),
             ("GFN2-xTB/ddCOSMO", "GFN2_COSMO"),
             ("GFN1-xTB/ddCOSMO", "GFN1_COSMO"),
             ("GFN1-xTB/ALPB", "GFN1_ALPB"), ("GFN1-xTB/GBSA", "GFN1_GBSA"),
             ("PM6-D3/COSMO", "PM6_D3_COSMO"),
             ("PM6-D3H4/COSMO", "PM6_D3H4_COSMO_NOFC")]
    keep = [(l, h) for l, h in pairs
            if f"{h}_even10" in m.index and f"{h}_basin_aa_unw" in m.index]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.95))
    y = np.arange(len(keep))
    panels = [("rep_spread", "range across trajectories\n(lower is better)", "a"),
              ("pool5_r2", "pooled $R^2$\n(higher is better)", "b")]
    for ax, (col, xlab, letter) in zip(axes, panels):
        a0 = [m.loc[f"{h}_even10", col] for _, h in keep]
        a1 = [m.loc[f"{h}_basin_aa_unw", col] for _, h in keep]
        span = max(a0 + a1) - min(a0 + a1)
        for i, (v0, v1) in enumerate(zip(a0, a1)):
            if abs(v1 - v0) < 0.06 * span:      # arrow would hide under the markers
                continue
            good = (v1 < v0) if col == "rep_spread" else (v1 > v0)
            ax.annotate("", xy=(v1, i), xytext=(v0, i),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3,
                                        color=BLUE if good else ORANGE,
                                        shrinkA=3, shrinkB=3))
        ax.plot(a0, y, "o", mfc="white", mec=GREY, mew=1.0, ms=6.2,
                clip_on=False, zorder=3)
        ax.plot(a1, y, "D", mfc=BLUE, mec="white", mew=0.6, ms=5,
                clip_on=False, zorder=4)
        lo, hi = min(a0 + a1), max(a0 + a1)
        pad = 0.14 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(y)
        ax.set_yticklabels([l for l, _ in keep] if ax is axes[0] else [],
                           fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(keep) - 0.45, -0.55)
        ax.set_xlabel(xlab)
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=2)
    handles = [
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec=GREY,
               mew=1.0, ms=5.5, label="even-10"),
        Line2D([0], [0], marker="D", ls="none", color=BLUE, ms=5,
               label="basin centroids"),
        Line2D([0], [0], color=BLUE, lw=1.3, marker=">", ms=4.5,
               label="improves"),
        Line2D([0], [0], color=ORANGE, lw=1.3, marker=">", ms=4.5,
               label="worsens"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.62, 0.925), borderaxespad=0,
               handletextpad=0.35, columnspacing=1.0, fontsize=7.4)
    # Slightly wider left gutter for the ddCOSMO y labels.
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.26, right=0.98, wspace=0.16)
    save(fig, "figure4.png")


# Figure 5: GFN2-xTB with four continuum solvents
def figure5_solvents():
    """Pooled R2 of GFN2-xTB with each continuum model and sampling protocol (a);
    offsets of ALPB, CPCM-X and ddCOSMO from GBSA against cross-interface salt
    bridges (b-d); experimental binding free energy against salt bridges (e)."""
    from scipy import stats
    metrics = pd.read_csv(REPO / "cpcmx_metrics.csv")
    systems = pd.read_csv(REPO / "solvent_offsets_systems.csv", index_col=0)
    systems["off_alpb"] = systems.gfn2_alpb - systems.gfn2_gbsa
    sets = [("iface", "Interface side-chain\nRMSD basins"),
            ("aa", "All-atom RMSD\nbasins"), ("bb", "Backbone RMSD\nbasins"),
            ("contacts", "Interchain-contact\nbasins"), ("even10", "Even-10")]
    methods = [("GFN2-xTB/GBSA", BLUE), ("GFN2-xTB/ALPB", GREEN),
               ("GFN2-xTB/CPCM-X", ORANGE), ("GFN2-xTB/ddCOSMO", GREY)]

    fig = plt.figure(figsize=(DOUBLE, 5.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.62)
    ax1 = fig.add_subplot(outer[0])
    bottom = outer[1].subgridspec(1, 2, width_ratios=[3.0, 1.05], wspace=0.34)
    trio = bottom[0].subgridspec(1, 3, wspace=0.10)
    width = 0.2
    x = np.arange(len(sets))
    for k, (meth, col) in enumerate(methods):
        r2s, lo, hi = [], [], []
        for s, _ in sets:
            row = metrics[(metrics.set == s) & (metrics.method == meth)
                          & (metrics.weight == "mean")].iloc[0]
            per = [float(v) for v in row.per_rep.split(" / ")]
            r2s.append(row.pool5_r2)
            lo.append(min(per))
            hi.append(max(per))
        pos = x + (k - 1.5) * width
        ax1.bar(pos, r2s, width, color=col, label=meth.replace("GFN2-xTB/", ""))
        ax1.vlines(pos, lo, hi, color="black", lw=0.6)
        ax1.scatter(pos, lo, marker="_", s=12, color="black", lw=0.6)
        ax1.scatter(pos, hi, marker="_", s=12, color="black", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lab for _, lab in sets])
    ax1.set_ylim(0.4, 0.82)
    ax1.set_ylabel("$R^2$ vs experiment", fontsize=8.5)
    ax1.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.0, columnspacing=1.2, borderaxespad=0.2, fontsize=8)
    ax1.tick_params(axis="y", labelsize=7.8)
    ax1.tick_params(axis="x", labelsize=7.4)
    ax1.text(-0.075, 1.10, "a", transform=ax1.transAxes, fontweight="bold", fontsize=10)

    xs = systems.n_saltbridge_x.to_numpy(float)
    grid = np.linspace(0, xs.max(), 10)
    trio_axes = []
    for j, (col, name, key, letter) in enumerate((
            (GREEN, "ALPB", "off_alpb", "b"), (ORANGE, "CPCM-X", "off_cpcmx", "c"),
            (GREY, "ddCOSMO", "off_cosmo", "d"))):
        ax = fig.add_subplot(trio[j], sharey=trio_axes[0] if trio_axes else None)
        trio_axes.append(ax)
        ys = systems[key].to_numpy(float)
        ys = ys - ys.mean()                  # a constant offset cannot change R2
        fit = stats.linregress(xs, ys)
        ax.axhline(0, color="black", lw=0.4)
        ax.scatter(xs, ys, s=13, color=col, edgecolor="none", zorder=3)
        ax.plot(grid, fit.intercept + fit.slope * grid, color=col, lw=1.0)
        ax.set_title(f"{name} $-$ GBSA", fontsize=8, pad=3)
        ax.text(0.04, 0.97, f"{fit.slope:.1f} kcal mol$^{{-1}}$ per bridge\n"
                f"r = {fit.rvalue:.2f}", transform=ax.transAxes, fontsize=6.8,
                va="top", ha="left", linespacing=1.25)
        ax.set_xlim(-0.25, 3.75)
        ax.set_xticks([0, 1, 2, 3])
        ax.tick_params(labelsize=7.4)
        ax.text(-0.10 if j else -0.42, 1.13, letter, transform=ax.transAxes,
                fontweight="bold", fontsize=10)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
    trio_axes[0].set_ylim(-30, 46)
    trio_axes[0].set_ylabel("Difference from GBSA\n(kcal mol$^{-1}$)", fontsize=8)
    trio_axes[1].set_xlabel("Salt bridges across the interface per frame", fontsize=8)
    cp = trio_axes[1]
    ys_cp = systems.off_cpcmx - systems.off_cpcmx.mean()
    for s, dx, dy, ha in (("1EMV", 5, -3, "left"), ("1AY7", -5, 2, "right"),
                          ("2HLE", 0, 5, "center")):
        cp.annotate(s, (systems.loc[s, "n_saltbridge_x"], ys_cp[s]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.8,
                    color="#7b3410", ha=ha)

    ax5 = fig.add_subplot(bottom[1])
    ye = systems.exp.to_numpy(float)
    r_exp = stats.pearsonr(xs, ye)
    ax5.scatter(xs, ye, s=13, color="#333333", edgecolor="none")
    ax5.set_title("Experiment", fontsize=8, pad=3)
    ax5.text(0.04, 0.03, f"r = {r_exp[0]:.2f}\np = {r_exp[1]:.2f}",
             transform=ax5.transAxes, fontsize=6.8, va="bottom", ha="left",
             linespacing=1.25)
    ax5.set_xlim(-0.25, 3.75)
    ax5.set_xticks([0, 1, 2, 3])
    ax5.set_ylim(-20, -4)
    ax5.set_ylabel("Experimental $\\Delta G$ (kcal mol$^{-1}$)", fontsize=8)
    ax5.set_xlabel("Salt bridges per frame", fontsize=8)
    ax5.tick_params(labelsize=7.4)
    ax5.text(-0.40, 1.13, "e", transform=ax5.transAxes, fontweight="bold", fontsize=10)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.085)
    save(fig, "figure5.png")


# Figure 6: ranking under resampling of the complexes
def _protocol_label(arm):
    """Full (non-compact) protocol name, as in Table S2."""
    return protocol_display_name(arm, compact=False)


def figure6_ranking():
    """Share of resamples of the twenty complexes in which each method family (a)
    and each of the most frequent individual protocols (b) ranks first."""
    fam = pd.read_csv(REPO / "ranking_bootstrap_family.csv")
    summ = pd.read_csv(REPO / "ranking_bootstrap_summary.csv",
                       index_col=0)["value"]
    sel = pd.read_csv(REPO / "ranking_bootstrap_top.csv").head(8)
    cpx_share = 100 * float(summ["cpcmx_share"])

    def colour(arm):
        if arm.startswith("MMGBSA"):
            return GREEN
        return ORANGE if arm.startswith("GFN2_CPCMX") else BLUE

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(6.6, 4.9), gridspec_kw={"height_ratios": [5, 8.4]})

    y = np.arange(len(fam))
    shares = 100 * fam.win_fraction.to_numpy(float)
    fam_col = {"GFN2-xTB": BLUE, "MM/GBSA": GREEN}
    axa.barh(y, shares, color=[fam_col.get(f, GREY) for f in fam.family],
             edgecolor="white", lw=.5, height=.68)
    i_gfn2 = int(np.where(fam.family == "GFN2-xTB")[0][0])
    axa.barh(i_gfn2, cpx_share, left=shares[i_gfn2] - cpx_share, color=ORANGE,
             edgecolor="white", lw=.5, height=.68)
    for i, (v, nprot, best) in enumerate(zip(shares, fam.n_arms, fam.best_full_r2)):
        axa.text(v + 0.8, i, f"{v:.1f}%   (best full-set $R^2$ {best:.3f})",
                 va="center", fontsize=7)
    axa.set_yticks(y)
    axa.set_yticklabels([f"{f}, {n} protocols" for f, n in zip(fam.family, fam.n_arms)],
                        fontsize=7.2)
    axa.set_ylim(len(fam) - 0.45, -0.6)
    axa.set_xlim(0, 100)
    axa.set_xlabel("resamples in which the family ranks first (%)", fontsize=7.8)
    axa.tick_params(labelsize=7.2)
    axa.text(-0.47, 1.06, "a", transform=axa.transAxes, fontweight="bold", fontsize=10)

    yb = np.arange(len(sel))
    axb.barh(yb, 100 * sel.win_fraction, color=[colour(a) for a in sel.arm],
             edgecolor="white", lw=.5, height=.68)
    for i, (v, rr) in enumerate(zip(sel.win_fraction, sel.full_r2)):
        axb.text(100 * v + .25, i, f"{100*v:.1f}%   (full-set $R^2$ {rr:.3f})",
                 va="center", fontsize=7)
    axb.set_yticks(yb)
    axb.set_yticklabels([_protocol_label(a).replace(", ", "\n", 1) for a in sel.arm],
                        fontsize=6.6, linespacing=1.05)
    axb.set_ylim(len(sel) - 0.45, -0.6)
    axb.set_xlim(0, 22)
    axb.set_xlabel("resamples in which the protocol ranks first (%)", fontsize=7.8)
    axb.tick_params(labelsize=7.2)
    axb.text(-0.47, 1.04, "b", transform=axb.transAxes, fontweight="bold", fontsize=10)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white") for c in (BLUE, ORANGE, GREEN, GREY)]
    axa.legend(h, ["GFN2-xTB with GBSA, ALPB or ddCOSMO", "GFN2-xTB/CPCM-X",
                   "Nwat-MM/GBSA", "GFN1-xTB, PM6, PM7"],
               frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.46),
               handlelength=1.2, fontsize=6.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.96, bottom=0.09, hspace=0.42)
    save(fig, "figure6.png")


# Figure 7: calibration against experiment
def figure7_calibration():
    pairs = [("GFN2_GBSA_basin_iface_sc_unw", "GFN2-xTB/GBSA, interface basins"),
             ("MMGBSA_Nwat30", "Nwat-MM/GBSA, 30 waters")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.35), sharey=True)
    for ax, (arm, title), letter in zip(axes, pairs, "ab"):
        d = sc[[arm]].join(exp).dropna()
        x, y = d[arm].to_numpy(float), d["Exp_dG"].to_numpy(float)
        sl, ic = np.polyfit(x, y, 1)
        ax.scatter(sl * x + ic, y, s=26, facecolor=BLUE, edgecolor="white",
                   linewidth=.5, zorder=3)
        lo, hi = y.min() - 1.2, y.max() + 1.2
        ax.plot([lo, hi], [lo, hi], ls=(0, (4, 2)), color=GREY, lw=.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{title}\npooled $R^2$ = {r2(arm):.3f}", fontsize=8, pad=4)
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", zorder=4)
        ax.set_xlabel("calibrated score (kcal/mol)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("experimental $\\Delta G$ (kcal/mol)")
    fig.tight_layout()
    save(fig, "figure7.png")


# Figure 8: standardised residuals
def figure8_residuals():
    main = [("MM/GBSA", "MMGBSA_Nwat30"),
            ("PM6-D3H4", "PM6_D3H4_COSMO_NOFC_even10"),
            ("GFN2/ALPB", "GFN2_ALPB_even10"),
            ("GFN2/GBSA", "GFN2_GBSA_even10"),
            ("GFN2/GBSA\nbasins aa", "GFN2_GBSA_basin_aa_unw"),
            ("GFN2/GBSA\nbasins iface", "GFN2_GBSA_basin_iface_sc_unw"),
            ("GFN2/CPCM-X\nbasins iface", "GFN2_CPCMX_basin_iface_sc_unw")]
    Z = {}
    for lab, a in main:
        d = sc[[a]].join(exp).dropna()
        x, y = d[a].to_numpy(float), d["Exp_dG"].to_numpy(float)
        sl, ic = np.polyfit(x, y, 1)
        res = y - (sl * x + ic)
        Z[lab] = pd.Series(res / res.std(ddof=1), index=d.index)
    z = pd.DataFrame(Z)
    z = z.loc[z.abs().mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    v = float(np.nanmax(np.abs(z.values)))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    # numeric annotation so the map is readable without colour
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z.values[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(val) > 0.62 * v else "#222222")
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels(
        ["MM/GBSA", "PM6-D3H4\neven-10", "GFN2/ALPB\neven-10",
         "GFN2/GBSA\neven-10", "GFN2/GBSA\nall-atom\nbasins",
         "GFN2/GBSA\ninterface\nbasins", "GFN2/CPCM-X\ninterface\nbasins"],
        fontsize=6.6)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels(z.index, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label("standardised residual", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "figure8.png")


for f in (figure1_overview, figure2_even10, figure3_spread,
          figure4_clustering_gain, figure5_solvents,
          figure6_ranking, figure7_calibration, figure8_residuals):
    f()
    print(" ", f.__name__)
print(f"wrote figures to {OUT}")
