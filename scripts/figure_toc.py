#!/usr/bin/env python3
"""Table-of-contents graphic (3.3 x 1.8 in) from protocol_metrics.csv and protocol_scores.csv.

Variants: 'benchmark' ranks the published methods on identical complexes and trajectories
against the Nwat-MM/GBSA reference; 'map' places every corrected protocol by accuracy and
by spread between trajectories. Uncorrected PM6 is excluded from both.
Usage: python figure_toc.py [--variant benchmark|map|all] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cpcmx_arms

from paths import DATA, FIGURES

REPO = DATA
FIGS = FIGURES

BLUE = "#1F5C99"
LIGHT_BLUE = "#8FB8DC"
MID_BLUE = "#4A82BE"
ORANGE = "#C0561A"
LIGHT_ORANGE = "#E09468"
GREEN = "#1E7A52"
GREY = "#8A94A0"
MID_GREY = "#6C7683"
DARK = "#16222E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 600,
})

M = cpcmx_arms.load_metrics(pd.read_csv(REPO / "protocol_metrics.csv").set_index("arm"))
S = cpcmx_arms.load_scores(pd.read_csv(REPO / "protocol_scores.csv").set_index("System"))
EXP = S["Exp_dG"].astype(float)

# PM6 without an empirical correction, excluded from the plots.
UNCORRECTED_PM6 = M.index.str.match(r"PM6_COSMO2?_")


def r2(arm: str) -> float:
    return float(M.loc[arm, "pool5_r2"])


def per_replica(arm: str) -> list[float]:
    """Single-trajectory R2 values of a protocol."""
    return [float(v) for v in str(M.loc[arm, "per_rep_r2"]).split("/")]


def replica_range(arm: str) -> tuple[float, float]:
    vals = per_replica(arm)
    return min(vals), max(vals)


def calibrated(arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Scores placed on the experimental scale by a linear fit, as in Figure 7."""
    d = S[[arm]].join(EXP).dropna()
    x = d[arm].to_numpy(float)
    y = d["Exp_dG"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    return slope * x + intercept, y


BEST = "GFN2_GBSA_basin_iface_sc_unw"


def best_scatter(ax):
    x, y = calibrated(BEST)
    lo, hi = y.min() - 1.3, y.max() + 1.3
    ax.plot([lo, hi], [lo, hi], linestyle=(0, (3.5, 2)), color=GREY, linewidth=0.7)
    ax.scatter(x, y, s=11, color=BLUE, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([-15, -10, -5])
    ax.set_yticks([-15, -10, -5])
    ax.tick_params(labelsize=4.6, length=1.8, pad=1)
    ax.set_xlabel("predicted ΔG (kcal/mol)", fontsize=5.0, labelpad=1.0)
    ax.set_ylabel("experiment", fontsize=5.0, labelpad=1.0)
    ax.set_title("Top Bar against Experiment\nR² = 0.70", fontsize=5.6,
                 color=BLUE, fontweight="bold", pad=2.0, linespacing=1.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Benchmark variant
def build_benchmark(out: Path):
    methods = [
        ("PM6-ML / COSMO", "PM6_ML_COSMO_even10", LIGHT_ORANGE),
        ("PM7 / COSMO2", "PM7_COSMO2_even10", LIGHT_ORANGE),
        ("PM6-D3H4 / COSMO", "PM6_D3H4_COSMO_even10", ORANGE),
        ("GFN1-xTB / ddCOSMO", "GFN1_COSMO_even10", MID_GREY),
        ("PM6-D3 / COSMO", "PM6_D3_COSMO_even10", ORANGE),
        ("Nwat-MM/GBSA (reference)", "MMGBSA_Nwat30", GREEN),
        ("GFN2-xTB / GBSA", "GFN2_GBSA_even10", LIGHT_BLUE),
        ("GFN2-xTB / CPCM-X", "GFN2_CPCMX_even10", MID_BLUE),
        # Same energy model as GFN2-xTB / GBSA, scored on basin frames chosen by
        # clustering instead of a fixed stride; shown in the inset scatter.
        ("GFN2-xTB / GBSA\n+ basin frames", BEST, BLUE),
    ]
    methods.sort(key=lambda m: r2(m[1]))

    fig = plt.figure(figsize=(3.3, 1.8), facecolor="white")
    fig.text(0.5, 0.978, "Can Semiempirical Methods Rank Protein–Protein Affinity?",
             ha="center", va="top", fontsize=7.9, fontweight="bold", color=DARK)

    ax = fig.add_axes([0.27, 0.315, 0.375, 0.545])
    for i, (label, arm, colour) in enumerate(methods):
        value = r2(arm)
        ax.barh(i, value, height=0.62, color=colour, edgecolor="none", zorder=2)
        # Whisker: range of single-trajectory R2, not an error bar; the pooled value
        # can lie outside it.
        lo, hi = replica_range(arm)
        ax.plot([lo, hi], [i, i], color=DARK, linewidth=0.6, zorder=4,
                solid_capstyle="butt")
        for end in (lo, hi):
            ax.plot([end, end], [i - 0.17, i + 0.17], color=DARK, linewidth=0.6,
                    zorder=4)
        ax.text(max(value, hi) + 0.012, i, f"{value:.2f}", va="center", ha="left",
                fontsize=5.4, fontweight="bold", color=DARK)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[0] for m in methods], fontsize=5.2, linespacing=1.15)
    readable = {LIGHT_ORANGE: ORANGE, LIGHT_BLUE: BLUE}
    for tick, (_, _, colour) in zip(ax.get_yticklabels(), methods):
        tick.set_color(readable.get(colour, colour))
    ax.set_ylim(-0.65, len(methods) - 0.35)
    ax.set_xlim(0, 0.80)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", labelsize=4.8, length=1.8, pad=1)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("R² with experimental ΔG", fontsize=5.5, labelpad=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.0, 1.035, "identical complexes and trajectories; evenly spaced frames unless the label says otherwise",
            transform=ax.transAxes, fontsize=4.7, color=GREY, va="bottom")
    ax.text(0.99, 0.03, "whiskers: single\ntrajectories", transform=ax.transAxes,
            fontsize=4.6, color=GREY, ha="right", va="bottom", linespacing=1.15)

    best_scatter(fig.add_axes([0.775, 0.275, 0.195, 0.40]))

    sqm = M[(M.n_sys == 20) & ~M.index.str.startswith("MMGBSA")
            & ~M.index.str.match(r"PM6_COSMO2?_")]
    fig.text(0.5, 0.105, f"20 protein–protein complexes  ×  5 independent MD trajectories  ×  {len(sqm)} corrected semiempirical protocols",
             ha="center", va="bottom", fontsize=5.1, color=DARK)
    fig.text(0.5, 0.015, "the strongest semiempirical protocol matches the classical reference and cannot be shown to beat it",
             ha="center", va="bottom", fontsize=5.1, color=GREY)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# Map variant
def build_map(out: Path):
    sel = M[~UNCORRECTED_PM6]

    def family(arm: str) -> str:
        if arm.startswith("MMGBSA"):
            return "MM/GBSA reference"
        if arm.startswith("GFN2"):
            return "GFN2-xTB"
        if arm.startswith("GFN1"):
            return "GFN1-xTB"
        return "corrected PM6 / PM7"

    fam = pd.Series([family(a) for a in sel.index], index=sel.index)
    style = {
        "corrected PM6 / PM7": dict(color=ORANGE, marker="o", size=9),
        "GFN1-xTB": dict(color=MID_GREY, marker="^", size=9),
        "GFN2-xTB": dict(color=BLUE, marker="o", size=9),
        "MM/GBSA reference": dict(color=GREEN, marker="D", size=11),
    }

    fig = plt.figure(figsize=(3.3, 1.8), facecolor="white")
    fig.text(0.5, 0.975, "Can semiempirical methods rank protein–protein affinity?",
             ha="center", va="top", fontsize=7.6, fontweight="bold", color=DARK)

    ax = fig.add_axes([0.085, 0.215, 0.535, 0.615])
    for name, st in style.items():
        idx = sel.index[fam == name]
        ax.scatter(sel.loc[idx, "rep_spread"], sel.loc[idx, "pool5_r2"],
                   s=st["size"], marker=st["marker"], color=st["color"],
                   edgecolor="white", linewidth=0.25, alpha=0.88, zorder=3,
                   label=name)

    best, even = "GFN2_GBSA_basin_iface_sc_unw", "GFN2_GBSA_even10"
    ax.annotate("", xy=(M.loc[best, "rep_spread"], M.loc[best, "pool5_r2"]),
                xytext=(M.loc[even, "rep_spread"], M.loc[even, "pool5_r2"]),
                arrowprops=dict(arrowstyle="-|>", color=DARK, linewidth=0.7,
                                mutation_scale=6, shrinkA=2, shrinkB=2), zorder=4)
    ax.text(0.16, 0.745, "conformational\nselection", fontsize=4.8, color=DARK,
            ha="left", va="bottom", linespacing=1.15)

    ax.set_xlim(-0.01, 0.47)
    ax.set_ylim(0.0, 0.83)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(labelsize=4.8, length=1.8, pad=1)
    ax.set_xlabel("spread between independent MD trajectories  →  less reproducible",
                  fontsize=5.0, labelpad=1.5)
    ax.set_ylabel("R² with experiment", fontsize=5.4, labelpad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=4.8, loc="lower center", frameon=False, ncol=4,
              handletextpad=0.2, borderpad=0.0, columnspacing=0.7,
              bbox_to_anchor=(0.5, 1.0))

    panel = fig.add_axes([0.665, 0.215, 0.33, 0.615])
    panel.axis("off")
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    entries = [
        (ORANGE, "Implementation, not only method",
         "three PM6-D3H4 codes span 0.08\nin R²; GBSA vs ALPB, 0.005"),
        (BLUE, "The Hamiltonian outranks the solvent",
         "GFN2 − GFN1 = 0.25–0.32 on\nidentical interface structures"),
        (GREEN, "The best protocol ties the classical one",
         "0.70 vs 0.65, p = 0.74:\n20 complexes cannot separate them"),
    ]
    y = 0.97
    for colour, head, bodytext in entries:
        panel.plot([0.0, 0.0], [y - 0.245, y], color=colour, linewidth=1.4,
                   solid_capstyle="butt")
        panel.text(0.055, y, head, fontsize=5.2, color=colour,
                   fontweight="bold", va="top")
        panel.text(0.055, y - 0.085, bodytext, fontsize=4.9, color=DARK,
                   va="top", linespacing=1.25)
        y -= 0.335

    fig.text(0.5, 0.01, "accuracy and reproducibility rank the protocols differently",
             ha="center", va="bottom", fontsize=5.2, color=GREY)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


BUILDERS = {"benchmark": build_benchmark, "map": build_map}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(BUILDERS) + ["all"], default="all")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the single chosen variant here")
    args = ap.parse_args()

    if args.variant == "all":
        for name, fn in BUILDERS.items():
            path = FIGS / f"toc_{name}.png"
            fn(path)
            print(f"wrote {path}")
        return

    path = args.out or FIGS / f"toc_{args.variant}.png"
    BUILDERS[args.variant](path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
