"""Charts for paper/varuna-short.tex.

Every number here is transcribed from a committed artifact, named in the
SOURCE comment above each block, so a chart can never drift from the text.
Run from the repo root:

    python scripts/make_short_paper_figures.py

Writes vector PDFs into paper/figures/. Palette is the validated categorical
set (blue / orange / aqua / yellow) with recessive chrome; every low-contrast
fill carries a direct value label or is backed by a table in the paper.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")

# ---------------------------------------------------------------- palette ---
# Validated categorical slots (light surface #fcfcfb); assigned in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, RED = "#4a3aa7", "#e34948"
# Ordinal blue ramp (steps 250 / 400 / 550) for ordered stages.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

COL, FULL = 3.45, 7.16  # IEEE single- and double-column widths, inches

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def chrome(ax, axis="y"):
    """Recessive grid on the value axis only; two spines removed."""
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left" if axis == "y" else "bottom"].set_color(BASELINE)
    ax.tick_params(length=2, width=0.6)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ============================================================ Fig: worth ===
# SOURCE: paper Table I / artifacts/trivial_baselines.json (150 scenes, 15 areas)
def fig_csi_worth():
    regions = ["Patna\n(inland)", "Bengaluru\n(inland)", "Mumbai\n(coastal)",
               "Chennai\n(coastal)", "Karnataka\n(inland)", "All 15\nareas"]
    allwet = [0.054, 0.022, 0.019, 0.021, 0.021, 0.022]
    clim = [0.286, 0.577, 0.520, 0.500, 0.615, 0.545]
    sar = [0.214, 0.502, 0.426, 0.383, 0.532, 0.456]

    x = np.arange(len(regions))
    w = 0.26
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ax.bar(x - w, allwet, w * 0.92, color=BLUE, label="all-wet (zero knowledge)", zorder=3)
    ax.bar(x, clim, w * 0.92, color=ORANGE, label="climatology (no model, no rain)", zorder=3)
    ax.bar(x + w, sar, w * 0.92, color=AQUA, label="radar vs radar (empirical ceiling)", zorder=3)

    for xi, c in zip(x, clim):  # direct labels on the headline series
        ax.text(xi, c + 0.012, f"{c:.2f}", ha="center", va="bottom",
                fontsize=6, color=INK, fontweight="bold")

    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylim(0, 0.88)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_ylabel("Critical Success Index")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, borderpad=0.1,
              labelspacing=0.3, bbox_to_anchor=(-0.02, 1.03))
    save(fig, "f_csi_worth.pdf")


# ========================================================== Fig: methods ===
# SOURCE: paper Table II / csi_net/results/twin_protocol.json (18 storms)
def fig_method_ladder():
    rows = [
        ("random", 0.014, BLUE, None),
        ("all-wet", 0.029, BLUE, None),
        ("best static index (HAND, TWI)", 0.039, BLUE, None),
        ("physics twin (antecedent rain)", 0.054, ORANGE, None),
        ("climatology (no model, no rain)", 0.288, BLUE, None),
        ("learned U-Net (terrain only)", 0.288, AQUA, 0.007),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    errs = [r[3] if r[3] else 0 for r in rows]

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ax.barh(y, vals, 0.62, color=cols, zorder=3,
            xerr=errs, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.6))
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=6.2, color=INK)

    chrome(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.345)
    ax.set_xlabel("mean CSI on the identical 18 storms")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["no model", "physics solver", "learned"], frameon=False,
              loc="lower right", handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    save(fig, "f_method_ladder.pdf")


# ========================================================= Fig: transfer ===
# SOURCE: csi_net/results/calibration_report.json + csi_net/RESULTS.md 4.3-4.4
def fig_transfer():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.15))

    # (a) what you hold out decides the number
    ax = axes[0]
    names = ["one tile\n(14 areas)", "one region\n(14 areas)", "one region\n(15 areas)"]
    glob = [0.2698, 0.1305, 0.1474]
    cal = [0.2542, 0.1566, 0.1746]
    oracle = [0.3030, 0.1615, 0.1845]
    x = np.arange(3)
    w = 0.33
    ax.bar(x - w / 2 - 0.01, glob, w, color=BLUE, label="global threshold", zorder=3)
    ax.bar(x + w / 2 + 0.01, cal, w, color=ORANGE, label="prior-matched threshold", zorder=3)
    for xi, o in zip(x, oracle):
        ax.plot([xi - 0.25, xi + 0.25], [o, o], color=INK2, lw=0.9,
                ls=(0, (2.2, 1.6)), zorder=5,
                label="oracle threshold" if xi == 0 else None)
    for xi, (g, c, o) in enumerate(zip(glob, cal, oracle)):
        top = max(g, c, o) + 0.009  # one shared height per group, clear of the oracle rule
        ax.text(xi - 0.04, top, f"{g:.3f}", ha="right", va="bottom", fontsize=5.8, color=INK)
        ax.text(xi + 0.04, top, f"{c:.3f}", ha="left", va="bottom", fontsize=5.8, color=INK)
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.425)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.set_ylabel("leave-one-out CSI")
    ax.set_title("(a) holding out a tile is not\nholding out a city", color=INK, pad=3)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderpad=0.1,
              labelspacing=0.28, bbox_to_anchor=(-0.03, 1.03))

    # (b) per region, and where the free correction bites
    ax = axes[1]
    regs = ["Karnataka\n(inland)", "Chennai\n(coastal)", "Patna\n(inland)", "Mumbai\n(coastal)"]
    g = [0.2207, 0.1642, 0.1537, 0.0613]
    c = [0.2335, 0.1889, 0.1711, 0.1061]
    pct = ["+5.8%", "+15.0%", "+11.4%", "+73.1%"]
    x = np.arange(4)
    ax.bar(x - w / 2 - 0.01, g, w, color=BLUE, zorder=3)
    ax.bar(x + w / 2 + 0.01, c, w, color=ORANGE, zorder=3)
    for xi, (gi, ci, p) in enumerate(zip(g, c, pct)):
        ax.text(xi + w / 2 + 0.01, ci + 0.006, p, ha="center", va="bottom",
                fontsize=5.8, color=INK, fontweight="bold")
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(regs, fontsize=5.9)
    ax.set_ylim(0, 0.29)
    ax.set_ylabel("held-out region CSI")
    ax.set_title("(b) the correction helps most\nwhere the model is weakest", color=INK, pad=3)

    # (c) the coastal ladder
    ax = axes[2]
    stages = ["no coast\nin training", "one coast\nin training", "+ free threshold\ncorrection"]
    vals = [0.0333, 0.0613, 0.1061]
    x = np.arange(3)
    ax.bar(x, vals, 0.52, color=RAMP, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=6, color=INK,
                fontweight="bold")
    ax.annotate("", xy=(1.72, 0.0995), xytext=(0.28, 0.0395), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=0.9,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(1.0, 0.088, "3.2$\\times$", color=VIOLET, fontsize=7, fontweight="bold", ha="center")
    ax.text(1.0, 0.0755, "no bigger model,\nno finer grid", color=INK2, fontsize=5.8, ha="center")
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 0.125)
    ax.set_ylabel("held-out coastal Mumbai CSI")
    ax.set_title("(c) coverage, then arithmetic", color=INK, pad=3)

    fig.subplots_adjust(wspace=0.34)
    save(fig, "f_transfer.pdf")


# ======================================================== Fig: non-levers ===
# SOURCE: csi_net/RESULTS.md 3.1 (capacity), 3.2 (grid), 5.2 (rainfall arms)
def fig_nonlevers():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95))

    # (a) capacity
    ax = axes[0]
    params = np.array([2.0, 7.9, 31.4, 70.6])
    csi = np.array([0.2206, 0.2321, 0.2213, 0.2194])
    band = 0.0066  # one seed standard deviation, from the 3-seed LOSO run
    ax.fill_between([1.4, 95], csi.mean() - band, csi.mean() + band,
                    color=BLUE, alpha=0.16, lw=0, zorder=2, label="$\\pm$1 seed s.d.")
    ax.plot(params, csi, "-o", color=BLUE, lw=1.4, ms=4, zorder=4, label="held-out CSI")
    for p, c in zip(params, csi):
        ax.text(p, c + 0.012, f"{c:.3f}", ha="center", va="bottom", fontsize=5.8, color=INK)
    chrome(ax)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 110)
    ax.set_xticks(params)
    ax.set_xticklabels(["2.0 M", "7.9 M", "31.4 M", "70.6 M"], fontsize=5.5)
    ax.minorticks_off()
    ax.set_ylim(0, 0.30)  # zero baseline: the claim is flatness, not the wiggle
    ax.set_xlabel("parameters (35$\\times$ range)")
    ax.set_ylabel("CSI")
    ax.set_title("(a) capacity does nothing", color=INK, pad=3)
    ax.legend(frameon=False, loc="lower left", handlelength=1.1, borderpad=0.1, labelspacing=0.22)

    # (b) resolution -- the skill multiple, because raw CSI is not comparable across grids
    ax = axes[1]
    x = np.arange(2)
    mult = [9.7, 10.4]
    ax.bar(x, mult, 0.40, color=[BLUE, ORANGE], zorder=3)
    for xi, m in zip(x, mult):
        ax.text(xi, m + 0.15, f"{m:.1f}$\\times$", ha="center", va="bottom",
                fontsize=7, color=INK, fontweight="bold")
    ax.text(0.5, 12.9, "4$\\times$ the cells,\n7 % of a skill multiple", ha="center",
            va="top", fontsize=5.9, color=INK2, linespacing=1.35)
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(["60 m grid\n(16 k cells)", "30 m grid\n(66 k cells)"])
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0, 13.2)
    ax.set_ylabel("CSI $\\div$ all-wet")
    ax.set_title("(b) resolution does nothing", color=INK, pad=3)

    # (c) rainfall
    ax = axes[2]
    x = np.arange(3)
    rank = [0.184, 0.272, 0.313]
    err = [0.013, 0.035, 0.012]
    ax.bar(x, rank, 0.44, color=[RED, YELLOW, AQUA], zorder=3,
           yerr=err, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=2))
    for xi, (r, e) in enumerate(zip(rank, err)):
        ax.text(xi, r + e + 0.008, f"{r:.3f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(["real\nrain", "shuffled\nrain", "no rain\nat all"])
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("ranking quality")
    ax.set_title("(c) real rainfall is the worst\nof the three inputs", color=INK, pad=3)

    fig.subplots_adjust(wspace=0.36)
    save(fig, "f_nonlevers.pdf")


# ========================================================= Fig: planning ===
# SOURCE: artifacts/<area>/canal_plan.json (reduction_pct), strategy_results.json,
#         CANAL_RESULTS.md, costbenefit.json
def fig_planning():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.25),
                             gridspec_kw=dict(width_ratios=[1.05, 0.78, 1.5]))

    # (a) the Patna strategy ladder
    ax = axes[0]
    names = ["excavation (8 sites)", "detention pits", "sparse canals",
             "727 storage basins", "drain spiderweb"]
    cut = [4.1, 7.8, 20.8, 30.0, 80.9]
    cols = [BLUE, BLUE, BLUE, BLUE, ORANGE]
    y = np.arange(5)
    ax.barh(y, cut, 0.62, color=cols, zorder=3)
    for yi, c in zip(y, cut):
        ax.text(c + 1.8, yi, f"{c:.1f}%", va="center", ha="left", fontsize=6.2, color=INK,
                fontweight="bold" if c > 80 else "normal")
    chrome(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 104)
    ax.set_xlabel("street flooding removed (%)")
    ax.set_title("(a) five strategies, one city,\nmeasured by re-simulation", color=INK, pad=3)

    # (b) cost per cubic metre actually removed -- density is cheaper, not dearer
    ax = axes[1]
    names = ["drain\nspiderweb", "sparse\ncanals", "distributed\nstorage"]
    cost = [850, 1309, 9200]
    x = np.arange(3)
    ax.bar(x, cost, 0.46, color=[ORANGE, BLUE, BLUE], zorder=3)
    for xi, c in zip(x, cost):
        ax.text(xi, c + 200, f"{c:,}", ha="center", va="bottom", fontsize=6.2, color=INK,
                fontweight="bold" if c == 850 else "normal")
    chrome(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 11200)
    ax.set_yticks([0, 2500, 5000, 7500, 10000])
    ax.set_yticklabels(["0", "2.5 k", "5 k", "7.5 k", "10 k"])
    ax.set_ylabel("INR per m$^3$ removed")
    ax.set_title("(b) the densest plan is also\nthe cheapest per m$^3$", color=INK, pad=3)

    # (c) every built area
    ax = axes[2]
    data = [("Patna-East", 89.4, "P"), ("Kolar", 83.8, "K"), ("Patna", 80.9, "P"),
            ("Patna-West", 80.3, "P"), ("Bengaluru", 72.6, "B"), ("Mumbai-S", 63.0, "M"),
            ("Ramanagara", 60.8, "K"), ("Chitradurga", 50.7, "K"), ("Mumbai-E", 50.2, "M"),
            ("Doddaballapura", 45.1, "K"), ("Mumbai-W", 39.2, "M"), ("Mumbai-N", 32.5, "M"),
            ("Chamarajanagara", 25.8, "K"), ("Chikkaballapur", 24.1, "K")]
    cmap = {"P": BLUE, "B": ORANGE, "M": AQUA, "K": YELLOW}
    y = np.arange(len(data))[::-1]
    ax.barh(y, [d[1] for d in data], 0.66, color=[cmap[d[2]] for d in data], zorder=3)
    for yi, d in zip(y, data):
        ax.text(d[1] + 1.2, yi, f"{d[1]:.1f}", va="center", ha="left", fontsize=5.8, color=INK)
    ax.axvspan(20.8, 25.7, color=MUTED, alpha=0.17, lw=0, zorder=2)
    ax.annotate("sparse-canal band\n(21--26 %)", xy=(23.2, len(data) - 0.4),
                xytext=(40, len(data) + 0.35), ha="left", va="center", fontsize=5.8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                            shrinkA=1, shrinkB=1))
    chrome(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.set_ylim(-0.7, len(data) + 0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("street flooding removed by the spiderweb (%)")
    ax.set_title("(c) the same planner on all fourteen built areas", color=INK, pad=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in "PBMK"]
    ax.legend(handles, ["Patna family", "Bengaluru", "Mumbai tiles", "Karnataka towns"],
              frameon=False, loc="lower right", handlelength=1.0, borderpad=0.2,
              labelspacing=0.22)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "f_planning.pdf")


# ========================================================= Fig: recharge ===
# SOURCE: artifacts/<area>/recharge_plan.json, last point of the dose curve
def fig_recharge():
    # (area, recharge share of runoff %, flood cut %, ground class)
    data = [("Chikkaballapur", 16.5, 62.7, "K"), ("Kolar", 14.1, 46.9, "K"),
            ("Doddaballapura", 10.0, 85.4, "K"), ("Chamarajanagara", 8.1, 88.2, "K"),
            ("Chitradurga", 8.0, 49.2, "K"), ("Ramanagara", 6.3, 76.8, "K"),
            ("Patna-East", 3.3, 38.4, "F"), ("Patna-West", 2.6, 22.4, "F"),
            ("Bengaluru", 1.6, 4.3, "B"), ("Mumbai-N", 1.1, 6.2, "M"),
            ("Patna", 0.7, 9.1, "F"), ("Mumbai-E", 0.2, 14.4, "M"),
            ("Mumbai-W", 0.2, 7.6, "M"), ("Mumbai-S", 0.0, 3.7, "M")]
    cmap = {"K": AQUA, "F": ORANGE, "B": YELLOW, "M": BLUE}
    names = [d[0] for d in data]
    y = np.arange(len(data))[::-1]
    cols = [cmap[d[3]] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))

    ax = axes[0]
    ax.barh(y, [d[1] for d in data], 0.66, color=cols, zorder=3)
    for yi, d in zip(y, data):
        ax.text(d[1] + 0.25, yi, f"{d[1]:.1f}", va="center", ha="left", fontsize=5.8, color=INK)
    # one hairline where the ground type changes -- plateau above, plain and coast below
    ax.axhline(y[6] + 0.5, color=MUTED, lw=0.6, ls=(0, (2.5, 2)), zorder=4)
    ax.text(19.2, y[3], "aquifer mined,\nstorage is real", ha="right", va="center",
            fontsize=5.8, color=INK2, linespacing=1.35)
    ax.text(19.2, y[11], "soil column already\nfull, nowhere to put it", ha="right",
            va="center", fontsize=5.8, color=INK2, linespacing=1.35)
    chrome(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 19.5)
    ax.set_xlabel("share of the storm's runoff banked underground (%)")
    ax.set_title("(a) metered recharge, not assumed", color=INK, pad=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in "KFBM"]
    ax.legend(handles, ["Karnataka plateau", "Gangetic flood plain", "Bengaluru", "Mumbai coast"],
              frameon=False, loc="lower right", handlelength=1.0, borderpad=0.2,
              labelspacing=0.25, columnspacing=1.0, ncol=2,
              bbox_to_anchor=(1.01, -0.015))

    ax = axes[1]
    ax.barh(y, [d[2] for d in data], 0.66, color=cols, zorder=3)
    for yi, d in zip(y, data):
        ax.text(d[2] + 1.2, yi, f"{d[2]:.1f}", va="center", ha="left", fontsize=5.8, color=INK)
    chrome(ax, axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 104)
    ax.set_xlabel("street flooding removed by the same structures (%)")
    ax.set_title("(b) no trade between flood control and water supply", color=INK, pad=3)

    fig.subplots_adjust(wspace=0.44)
    save(fig, "f_recharge.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_csi_worth()
    fig_method_ladder()
    fig_transfer()
    fig_nonlevers()
    fig_planning()
    fig_recharge()
