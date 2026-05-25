"""
fig_violin_cvar.py
------------------
Violin plot comparing nominal vs robust OOS profit distributions,
with CVaR 5% highlighted inside each violin body.

Run from experiments/:
    python fig_violin_cvar.py
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.3,
})

EXCEL = "/root/.claude/uploads/6ba662e2-d046-4036-9d5e-25bc2d3887f1/a8166f9f-05_sensitivity_results.xlsx"

# ── Palette ───────────────────────────────────────────────────────────────────
NOM_LIGHT  = "#6baed6"   # light blue
NOM_DARK   = "#08519c"   # dark blue  (CVaR tail + line)
ROB_LIGHT  = "#fb6a4a"   # light red
ROB_DARK   = "#a50f15"   # dark red   (CVaR tail + line)

# ── Helpers ───────────────────────────────────────────────────────────────────
def cvar5(arr):
    k = max(1, int(np.ceil(0.05 * len(arr))))
    return float(np.mean(np.sort(arr)[:k]))

def pct5(arr):
    return float(np.percentile(arr, 5))


def draw_violin(ax, data, x_center, half_width,
                light_color, dark_color, label=None,
                bw_method="scott"):
    """
    Draw a single violin with:
      - full KDE body in light_color
      - bottom 5% of distribution shaded in dark_color
      - CVaR 5% shown as a bold horizontal bar
      - median as a white bar
      - mean as a diamond marker
    """
    data = np.asarray(data, dtype=float)
    n    = len(data)

    # KDE over a slightly extended range
    lo = np.min(data) - 0.05 * np.ptp(data)
    hi = np.max(data) + 0.05 * np.ptp(data)
    y_grid   = np.linspace(lo, hi, 600)
    kde      = gaussian_kde(data, bw_method=bw_method)
    density  = kde(y_grid)

    # Scale so max half-width = half_width
    scale    = half_width / density.max()
    d_scaled = density * scale

    # ── Full violin body ──────────────────────────────────────────────────────
    ax.fill_betweenx(y_grid,
                     x_center - d_scaled, x_center + d_scaled,
                     color=light_color, alpha=0.55, linewidth=0,
                     label=label, zorder=2)
    # Thin outline
    ax.plot(x_center - d_scaled, y_grid, color=light_color,
            lw=0.8, alpha=0.9, zorder=2)
    ax.plot(x_center + d_scaled, y_grid, color=light_color,
            lw=0.8, alpha=0.9, zorder=2)

    # ── CVaR 5%: shade the bottom-5% tail ────────────────────────────────────
    p5   = pct5(data)
    cv5  = cvar5(data)

    tail_mask = y_grid <= p5
    if tail_mask.any():
        ax.fill_betweenx(y_grid[tail_mask],
                         x_center - d_scaled[tail_mask],
                         x_center + d_scaled[tail_mask],
                         color=dark_color, alpha=0.85, linewidth=0, zorder=3)

    # Dashed boundary at 5th percentile (VaR)
    if tail_mask.any():
        d_at_p5 = float(kde(np.array([p5]))[0] * scale)
        ax.hlines(p5, x_center - d_at_p5, x_center + d_at_p5,
                  colors=dark_color, lw=1.2, linestyle="--", zorder=4)

    # Bold bar at CVaR value
    d_at_cv5 = float(kde(np.array([cv5]))[0] * scale)
    ax.hlines(cv5, x_center - d_at_cv5, x_center + d_at_cv5,
              colors=dark_color, lw=3.0, zorder=5)

    # Annotate CVaR value (right side, small)
    ax.annotate(f"CVaR$_{{5\\%}}$\n{cv5/1e3:.1f}k",
                xy=(x_center + d_at_cv5, cv5),
                xytext=(x_center + d_at_cv5 + 0.06, cv5),
                fontsize=6.5, color=dark_color,
                va="center", ha="left",
                annotation_clip=False)

    # ── Median (white bar) ────────────────────────────────────────────────────
    med      = np.median(data)
    d_at_med = float(kde(np.array([med]))[0] * scale)
    ax.hlines(med, x_center - d_at_med, x_center + d_at_med,
              colors="white", lw=2.2, zorder=6)

    # ── Mean (diamond) ────────────────────────────────────────────────────────
    ax.scatter(x_center, np.mean(data),
               marker="D", s=22, color="white",
               edgecolors=dark_color, linewidths=1.0,
               zorder=7)

    return cv5


# ── Load OOS data ─────────────────────────────────────────────────────────────
oos = pd.read_excel(EXCEL, sheet_name="OOS_Raw")

# ── Configuration — easy to change ──────────────────────────────────────────
V_SCALE = 0.75
W_VAL   = 10
GAMMAS  = [1, 2, 3, 4]
COLS    = ("profit_a_nom", "profit_a_rob")
HALF_W  = 0.32          # half-width of each violin
GAP     = 0.08          # gap between the two violins in a pair
GROUP_W = 2.0           # x-distance between Gamma groups

sub = oos[(oos["v_scale"] == V_SCALE) & (oos["w"] == W_VAL)]

# ── Build figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

x_ticks, x_labels = [], []
cvar_summary = {}

for gi, gam in enumerate(GAMMAS):
    g = sub[sub["gamma"] == gam]
    arr_nom = g["profit_a_nom"].values
    arr_rob = g["profit_a_rob"].values

    x_base  = gi * GROUP_W
    x_nom   = x_base - (HALF_W + GAP / 2)
    x_rob   = x_base + (HALF_W + GAP / 2)

    lbl_nom = "Nominal $x$"  if gi == 0 else None
    lbl_rob = "Robust $x$"   if gi == 0 else None

    cv_nom = draw_violin(ax, arr_nom, x_nom, HALF_W,
                         NOM_LIGHT, NOM_DARK, label=lbl_nom)
    cv_rob = draw_violin(ax, arr_rob, x_rob, HALF_W,
                         ROB_LIGHT, ROB_DARK, label=lbl_rob)

    cvar_summary[gam] = (cv_nom, cv_rob)
    x_ticks.append(x_base)
    x_labels.append(f"$\\Gamma = {gam}$")

# ── Reference line at 0 ──────────────────────────────────────────────────────
ax.axhline(0, color="black", lw=0.9, linestyle=":", zorder=1, alpha=0.6)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.set_xlim(x_ticks[0] - GROUP_W * 0.7, x_ticks[-1] + GROUP_W * 0.7)
ax.set_ylabel("Out-of-sample profit")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
ax.grid(axis="y", linestyle="--")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=NOM_LIGHT, alpha=0.7, label="Nominal $x$  (Scen. A)"),
    mpatches.Patch(color=ROB_LIGHT, alpha=0.7, label="Robust $x$   (Scen. A)"),
]
legend_lines = [
    mpatches.Patch(color=NOM_DARK,  alpha=0.85, label="CVaR$_{5\\%}$ tail — Nominal"),
    mpatches.Patch(color=ROB_DARK,  alpha=0.85, label="CVaR$_{5\\%}$ tail — Robust"),
    plt.Line2D([0],[0], color="gray",  lw=2.2, label="Median"),
    plt.Line2D([0],[0], color="gray",  lw=1.2,
               linestyle="--", label="5th percentile (VaR)"),
    plt.Line2D([0],[0], marker="D", color="w",
               markeredgecolor="gray", markersize=5, label="Mean"),
]
ax.legend(handles=legend_patches + legend_lines,
          ncol=2, frameon=True, framealpha=0.92,
          loc="upper right", fontsize=8.5,
          edgecolor="0.8")

# ── Subtitle with parameters ──────────────────────────────────────────────────
ax.set_title(
    f"OOS Profit Distributions — Nominal vs. Robust  "
    f"($v = {V_SCALE},\\; w = {W_VAL}$, adaptive pricing)",
    fontsize=11, pad=8)

# ── Print CVaR summary ────────────────────────────────────────────────────────
print(f"\n{'Γ':>4}  {'CVaR5 Nom':>12}  {'CVaR5 Rob':>12}  {'Δ CVaR5':>12}")
print("-" * 46)
for gam, (cn, cr) in cvar_summary.items():
    print(f"{gam:>4}  {cn:>12,.0f}  {cr:>12,.0f}  {cr-cn:>+12,.0f}")

fig.tight_layout()
fig.savefig("fig_violin_cvar.pdf", bbox_inches="tight")
fig.savefig("fig_violin_cvar.png", dpi=300, bbox_inches="tight")
print("\nSaved fig_violin_cvar.pdf / .png")
plt.show()
