#!/usr/bin/env python3
"""Figure S8: decomposition of the CPCM-X offset, written to figures/figureS8.png.

(a) Conductor part, E(conductor) - E(GBSA), and post-SCF water terms of the CPCM-X
    offset from GFN2-xTB/GBSA against cross-interface salt bridges, mean-centred.
(b) Extra absolute residual of CPCM-X over GBSA per complex, each method fitted
    to experiment separately. Input: per-complex table from analyze_solvent_offsets.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import DATA, FIGURES

REPO = DATA
OUT = FIGURES
OUT.mkdir(exist_ok=True)

DOUBLE = 7.00
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300})
BLUE, ORANGE, GREY = "#2b6cb0", "#c05621", "#8a94a6"

s = pd.read_csv(REPO / "solvent_offsets_systems.csv", index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9),
                               gridspec_kw={"width_ratios": [1.0, 1.35]})
xs = s.n_saltbridge_x.to_numpy(float)
for key, col, mk, name in (("off_cond", ORANGE, "o", "conductor part"),
                           ("post_scf", GREY, "s", "post-SCF water terms")):
    ys = s[key].to_numpy(float)
    ys = ys - ys.mean()
    fit = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    ax1.scatter(xs, ys, s=16, marker=mk, color=col, edgecolor="white", lw=0.4,
                zorder=3, label=f"{name}: r = {r:.2f}, slope {fit[0]:.1f}")
    g = np.linspace(xs.min(), xs.max(), 20)
    ax1.plot(g, np.polyval(fit, g), color=col, lw=1.0, zorder=2)
ax1.axhline(0, color="#999999", lw=0.5, zorder=1)
ax1.set_xlabel("salt bridges per frame")
ax1.set_ylabel("part of the CPCM-X offset,\ncentred (kcal/mol)")
ax1.set_ylim(-32, 42)
ax1.legend(frameon=False, loc="upper left", fontsize=6.8, handletextpad=0.2,
           borderaxespad=0.2)
ax1.set_title("a", loc="left", fontweight="bold", fontsize=11, pad=4)

d = s.sort_values("extra_err_cpcmx", ascending=False)
x = np.arange(len(d))
vals = d.extra_err_cpcmx.to_numpy(float)
ax2.bar(x, vals, width=0.7, color=[ORANGE if v > 0 else BLUE for v in vals],
        edgecolor="white", lw=0.5)
for i, (v, n) in enumerate(zip(vals, d.n_saltbridge_x)):
    ax2.text(i, v + (0.12 if v >= 0 else -0.12), f"{n:.1f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=5.8, color="#333333")
ax2.axhline(0, color="#555555", lw=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(d.index, rotation=90, fontsize=6.8)
ax2.set_xlim(-0.6, len(d) - 0.4)
ax2.set_ylim(-3.8, 3.4)
ax2.set_ylabel("extra |residual| of CPCM-X\nover GBSA (kcal/mol)")
ax2.text(0.99, 0.97, "number on bar: salt bridges per frame",
         transform=ax2.transAxes, ha="right", va="top", fontsize=6.6,
         color="#333333")
ax2.set_title("b", loc="left", fontweight="bold", fontsize=11, pad=4)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "figureS8.png", bbox_inches="tight", dpi=300)
print(f"wrote {OUT / 'figureS8.png'}")
