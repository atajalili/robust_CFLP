"""
paper_figures.py
----------------
Generates paper-ready figures from 05_sensitivity_results.xlsx.

Run from the experiments/ directory:
    python paper_figures.py

Outputs (PDF + PNG):
    fig1_robustification.pdf/.png   -- robustification trade-off (2×2)
    fig2_vap.pdf/.png               -- value of adaptive pricing (2×2)
    fig3_combined.pdf/.png          -- combined gain + risk decomposition (1×3)
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Global style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
})

def _fmt(ax, ylabel=None, xlabel=None, ythou=True, ypct=False):
    """Apply common axis formatting."""
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ythou:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
    if ypct:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y")

def save(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved {stem}.pdf / .png")

# ── Load data ─────────────────────────────────────────────────────────────────
EXCEL = "/root/.claude/uploads/6ba662e2-d046-4036-9d5e-25bc2d3887f1/a8166f9f-05_sensitivity_results.xlsx"
df    = pd.read_excel(EXCEL, sheet_name="Summary")
oos   = pd.read_excel(EXCEL, sheet_name="OOS_Raw")

GAMMAS   = sorted(df["gamma"].unique())
VSCALES  = sorted(df["v_scale"].unique())
WS       = sorted(df["w"].unique())

# Colour palettes (colorblind-safe)
V_COLORS  = {0.50: "#1f77b4", 0.75: "#ff7f0e", 1.00: "#2ca02c", 1.25: "#d62728"}
W_COLORS  = {10:   "#1f77b4", 100:  "#ff7f0e", 1000: "#9467bd"}
G_COLORS  = {1:    "#1f77b4", 2:    "#ff7f0e", 3:    "#2ca02c", 4:    "#d62728"}
V_LABELS  = {v: f"$v={v}$" for v in VSCALES}
W_LABELS  = {w: f"$w={w}$" for w in WS}
G_LABELS  = {g: f"$\\Gamma={g}$" for g in GAMMAS}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Robustification Trade-off (2 × 2)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1 — Robustification Trade-off …")

fig1, axes = plt.subplots(2, 2, figsize=(9, 7))
sub_w10 = df[df["w"] == 10].copy()
sub_w10["PoR_pct"] = 100 * (sub_w10["nom_profit"] - sub_w10["rob_profit_LB"]) \
                     / sub_w10["nom_profit"]

# ── [0,0]  Price of Robustness (%) ──────────────────────────────────────────
ax = axes[0, 0]
for v in VSCALES:
    g = sub_w10[sub_w10["v_scale"] == v].sort_values("gamma")
    ax.plot(g["gamma"], g["PoR_pct"], "-o",
            color=V_COLORS[v], label=V_LABELS[v])

# Mark "no-facility" corner cases with ×
corner = sub_w10[sub_w10["rob_profit_LB"] < 1.0]
ax.scatter(corner["gamma"], corner["PoR_pct"],
           marker="x", s=60, color="black", zorder=5,
           linewidths=2, label="No facility opened")

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("Price of Robustness (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_title("(a) Price of Robustness", loc="left", fontweight="bold")
ax.legend(ncol=2, frameon=False)
ax.grid(axis="y")

# ── [0,1]  Worst-case robustification value ──────────────────────────────────
ax = axes[0, 1]
for v in VSCALES:
    g = sub_w10[sub_w10["v_scale"] == v].sort_values("gamma")
    # Replace degenerate rows with NaN so line breaks
    y = g["wc_rob_val_a"].where(g["rob_profit_LB"] > 1.0, other=np.nan)
    ax.plot(g["gamma"], y / 1e3, "-o",
            color=V_COLORS[v], label=V_LABELS[v])

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("Worst-case robustification value (\\$000s)")
ax.set_title("(b) Worst-Case Robustification Value", loc="left", fontweight="bold")
ax.legend(ncol=2, frameon=False)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# ── [1,0]  Average robustification value — trade-off view ───────────────────
ax = axes[1, 0]
ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=2)

for v in VSCALES:
    g = sub_w10[sub_w10["v_scale"] == v].sort_values("gamma")
    y = g["rob_val_a_mean"].where(g["rob_profit_LB"] > 1.0, other=np.nan)
    ax.plot(g["gamma"], y / 1e3, "-o",
            color=V_COLORS[v], label=V_LABELS[v])

# Shade benefit/cost regions
ylim_lo, ylim_hi = ax.get_ylim()
ylim_abs = max(abs(lim) for lim in ax.get_ylim())
ax.fill_between([0.8, 4.2], 0,  ylim_abs,
                color="#2ca02c", alpha=0.07, zorder=0)
ax.fill_between([0.8, 4.2], 0, -ylim_abs,
                color="#d62728", alpha=0.07, zorder=0)
ax.text(4.15, ylim_abs * 0.15, "Rob. better", color="#2ca02c", fontsize=8,
        ha="right", va="bottom")
ax.text(4.15, -ylim_abs * 0.15, "Nom. better", color="#d62728", fontsize=8,
        ha="right", va="top")

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("Avg. robustification value (\\$000s)")
ax.set_title("(c) Average-Case Robustification Value", loc="left", fontweight="bold")
ax.legend(ncol=2, frameon=False)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# ── [1,1]  CVaR5% — nominal vs robust ────────────────────────────────────────
ax = axes[1, 1]
# Focus on v=0.75 and v=1.25 for clarity; show both solutions
line_styles = {"nom": ("-", "o"), "rob": ("--", "s")}
for v, ls_key in [(0.75, None), (1.25, None)]:
    g = sub_w10[sub_w10["v_scale"] == v].sort_values("gamma")
    c = V_COLORS[v]
    ax.plot(g["gamma"], g["a_nom_cvar5"] / 1e3, "-o",
            color=c, alpha=0.6, label=f"Nominal, {V_LABELS[v]}")
    ax.plot(g["gamma"], g["a_rob_cvar5"] / 1e3, "--s",
            color=c, label=f"Robust,  {V_LABELS[v]}")

ax.axhline(0, color="black", linewidth=0.9, linestyle=":", zorder=2)
ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("CVaR$_{5\\%}$ of profit (\\$000s)")
ax.set_title("(d) Tail-Risk: CVaR$_{5\\%}$", loc="left", fontweight="bold")
ax.legend(ncol=2, frameon=False, fontsize=8)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# Add legend for solid=nominal, dashed=robust
extra = [
    Line2D([0],[0], color="gray", ls="-",  marker="o", ms=5, label="Nominal"),
    Line2D([0],[0], color="gray", ls="--", marker="s", ms=5, label="Robust"),
]
leg = ax.legend(handles=extra, loc="upper right", frameon=True,
                fontsize=8, framealpha=0.9)
ax.add_artist(leg)

fig1.tight_layout(pad=1.5)
save(fig1, "fig1_robustification")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Value of Adaptive Pricing (2 × 2)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2 — Value of Adaptive Pricing …")

fig2, axes = plt.subplots(2, 2, figsize=(9, 7))

sub_w10 = df[df["w"] == 10].copy()
sub_v75 = df[df["v_scale"] == 0.75].copy()

# Percentage VAP
df["vap_nom_pct"] = 100 * df["vap_nom_mean"] / df["a_nom_mean"].replace(0, np.nan)
df["vap_rob_pct"] = 100 * df["vap_rob_mean"] / df["a_rob_mean"].replace(0, np.nan)

# ── [0,0]  Average VAP% vs Γ (w=10, all v_scales) ──────────────────────────
ax = axes[0, 0]
sub = df[df["w"] == 10].copy()
for v in [0.75, 1.00, 1.25]:   # skip v=0.50 (degenerate)
    g = sub[sub["v_scale"] == v].sort_values("gamma")
    c = V_COLORS[v]
    # Only plot non-degenerate rows
    mask = g["rob_profit_LB"] > 1.0
    ax.plot(g.loc[mask, "gamma"], g.loc[mask, "vap_nom_pct"], "-o",
            color=c, alpha=0.65, label=f"Nominal, {V_LABELS[v]}")
    ax.plot(g.loc[mask, "gamma"], g.loc[mask, "vap_rob_pct"], "--s",
            color=c, label=f"Robust, {V_LABELS[v]}")

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("VAP / Scen-A profit (%)")
ax.set_title("(a) VAP as Share of Adaptive Profit  ($w=10$)",
             loc="left", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="y")
# Shared style legend
extra = [
    Line2D([0],[0], color="gray", ls="-",  marker="o", ms=5, label="Nominal"),
    Line2D([0],[0], color="gray", ls="--", marker="s", ms=5, label="Robust"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + extra, frameon=False, ncol=2, fontsize=8)

# ── [0,1]  Average VAP% vs w (v=0.75, lines per Γ) ──────────────────────────
ax = axes[0, 1]
sub = df[df["v_scale"] == 0.75].copy()
sub["vap_rob_pct"] = 100 * sub["vap_rob_mean"] / sub["a_rob_mean"].replace(0, np.nan)
for g_val in GAMMAS:
    g = sub[sub["gamma"] == g_val].sort_values("w")
    mask = g["rob_profit_LB"] > 1.0
    ax.plot(np.log10(g.loc[mask, "w"]), g.loc[mask, "vap_rob_pct"], "-o",
            color=G_COLORS[g_val], label=G_LABELS[g_val])

ax.set_xticks(np.log10(WS))
ax.set_xticklabels([str(w) for w in WS])
ax.set_xlabel("Congestion cost $w$ (log scale)")
ax.set_ylabel("VAP / Scen-A profit (%)")
ax.set_title("(b) VAP (Robust) vs. Congestion Cost  ($v=0.75$)",
             loc="left", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(frameon=False)
ax.grid(axis="y")

# ── [1,0]  Worst-case VAP — nominal vs robust (w=10, v=0.75) ─────────────────
ax = axes[1, 0]
sub = df[(df["w"] == 10) & (df["v_scale"] == 0.75)].sort_values("gamma")
ax.bar(np.array(GAMMAS) - 0.2, sub["wc_vap_nom"] / 1e3, 0.38,
       color="#1f77b4", alpha=0.85, label="Nominal $x$", edgecolor="k", lw=0.5)
ax.bar(np.array(GAMMAS) + 0.2, sub["wc_vap_rob"] / 1e3, 0.38,
       color="#ff7f0e", alpha=0.85, label="Robust $x$",  edgecolor="k", lw=0.5)

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("Worst-case VAP (\\$000s)")
ax.set_title("(c) Worst-Case VAP  ($v=0.75,\\; w=10$)",
             loc="left", fontweight="bold")
ax.legend(frameon=False)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# ── [1,1]  Combined gain heatmap (v_scale × Γ at w=10) ─────────────────────
ax = axes[1, 1]
sub = df[df["w"] == 10].copy()
pivot = sub.pivot(index="v_scale", columns="gamma", values="combined_mean") / 1e3

vabs = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 1)
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vabs, vmax=vabs,
               aspect="auto", origin="lower")

ax.set_xticks(range(len(GAMMAS)))
ax.set_xticklabels([f"$\\Gamma={g}$" for g in GAMMAS])
ax.set_yticks(range(len(VSCALES)))
ax.set_yticklabels([f"$v={v}$" for v in VSCALES])
ax.set_xlabel("Uncertainty budget")
ax.set_ylabel("Willingness-to-pay scale")
ax.set_title("(d) Avg. Combined Gain: Robust+Adaptive vs. Nominal+Fixed (\\$000s)\n"
             "$w=10$", loc="left", fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.85, label="\\$000s")

for i in range(len(VSCALES)):
    for j in range(len(GAMMAS)):
        val = pivot.values[i, j]
        if not np.isfinite(val):
            txt = "—"
        elif abs(val) > 10:
            txt = f"{val:.0f}"
        else:
            txt = f"{val:.1f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if abs(val) > 0.6 * vabs else "black")

fig2.tight_layout(pad=1.5)
save(fig2, "fig2_vap")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Risk Decomposition across Γ (1 × 3)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3 — Risk Decomposition …")

fig3, axes = plt.subplots(1, 3, figsize=(13, 4.5))

sub_w10 = df[df["w"] == 10].copy()

# ── [0]  CVaR5% for all four streams (v=1.00, w=10) ─────────────────────────
ax = axes[0]
sub = sub_w10[sub_w10["v_scale"] == 1.0].sort_values("gamma")
STREAM_COLORS = {
    "a_nom": "#1f77b4", "b_nom": "#aec7e8",
    "a_rob": "#d62728", "b_rob": "#f4a582",
}
STREAM_LABELS = {
    "a_nom": "Nom + Adaptive", "b_nom": "Nom + Fixed-px",
    "a_rob": "Rob + Adaptive", "b_rob": "Rob + Fixed-px",
}
for key in ["a_nom", "b_nom", "a_rob", "b_rob"]:
    col = f"{key}_cvar5"
    ls  = "-o" if "rob" in key else "--o"
    ax.plot(sub["gamma"], sub[col] / 1e3, ls,
            color=STREAM_COLORS[key], label=STREAM_LABELS[key],
            alpha=0.9)

ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)
ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("CVaR$_{5\\%}$ (\\$000s)")
ax.set_title("(a) Tail Risk: CVaR$_{5\\%}$ by Stream\n"
             "$v=1.00,\\; w=10$", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# ── [1]  Probability of loss (v=0.75, all w) ──────────────────────────────
ax = axes[1]
sub = df[df["v_scale"] == 0.75].copy()
for w_val in WS:
    g = sub[sub["w"] == w_val].sort_values("gamma")
    ax.plot(g["gamma"], 100 * g["a_nom_prob_loss"], "-o",
            color=W_COLORS[w_val], label=f"Nominal, {W_LABELS[w_val]}", alpha=0.75)
    ax.plot(g["gamma"], 100 * g["a_rob_prob_loss"], "--s",
            color=W_COLORS[w_val], label=f"Robust, {W_LABELS[w_val]}", alpha=0.75)

ax.set_xticks(GAMMAS)
ax.set_xlabel("Uncertainty budget $\\Gamma$")
ax.set_ylabel("$P(\\pi < 0)$ (%)")
ax.set_title("(b) Probability of Loss\n$v=0.75$",
             loc="left", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
extra = [
    Line2D([0],[0], color="gray", ls="-",  marker="o", ms=5, label="Nominal"),
    Line2D([0],[0], color="gray", ls="--", marker="s", ms=5, label="Robust"),
]
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles[:3] + extra, ncol=2, frameon=False, fontsize=8)
ax.grid(axis="y")

# ── [2]  OOS profit box plots for a_nom and a_rob at v=1.00, w=10 ──────────
ax = axes[2]
sub_oos = oos[(oos["v_scale"] == 1.00) & (oos["w"] == 10)]
positions = []
data      = []
colors    = []
x_ticks   = []
x_labels  = []
gap = 0.55

for gi, gam in enumerate(GAMMAS):
    g = sub_oos[sub_oos["gamma"] == gam]
    p_nom = gi * 3 * gap
    p_rob = p_nom + gap
    positions += [p_nom, p_rob]
    data      += [g["profit_a_nom"].values, g["profit_a_rob"].values]
    colors    += ["#1f77b4", "#d62728"]
    x_ticks.append(p_nom + gap / 2)
    x_labels.append(f"$\\Gamma={gam}$")

bp = ax.boxplot(data, positions=positions, widths=0.45,
                patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1), capprops=dict(lw=1),
                flierprops=dict(marker=".", ms=3, alpha=0.4))
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.set_ylabel("OOS Profit (\\$000s)")
ax.set_title("(c) OOS Profit Distribution: Adaptive Pricing\n"
             "$v=1.00,\\; w=10$",
             loc="left", fontweight="bold")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
leg_patches = [
    mpatches.Patch(color="#1f77b4", alpha=0.75, label="Nominal $x$"),
    mpatches.Patch(color="#d62728", alpha=0.75, label="Robust $x$"),
]
ax.legend(handles=leg_patches, frameon=False)
ax.grid(axis="y")

fig3.tight_layout(pad=1.5)
save(fig3, "fig3_risk_decomposition")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Robustification + VAP Sensitivity (2 × 3 heatmaps)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4 — Sensitivity Heatmaps …")

fig4, axes = plt.subplots(2, 3, figsize=(13, 8))

# At v=0.75: heatmaps over w × Γ for 6 key metrics
sub075 = df[df["v_scale"] == 0.75].copy()
sub075["vap_rob_pct"] = 100 * sub075["vap_rob_mean"] / \
    sub075["a_rob_mean"].replace(0, np.nan)

metrics = [
    ("wc_rob_val_a",    "Worst-case Robustification\nValue (\\$000s)",    "RdYlGn",  False, 1e3),
    ("rob_val_a_mean",  "Avg. Robustification\nValue (\\$000s)",          "RdYlGn",  True,  1e3),
    ("a_rob_cvar5",     "CVaR$_{5\\%}$ — Robust+Adaptive\n(\\$000s)",    "YlGn",    False, 1e3),
    ("wc_vap_rob",      "Worst-case VAP\n— Robust $x$ (\\$000s)",         "Blues",   False, 1e3),
    ("vap_rob_pct",     "Avg. VAP as \\% of Adaptive\nProfit — Robust $x$","Purples",False, 1.0),
    ("combined_mean",   "Avg. Combined Gain:\nRob+Adaptive vs. Nom+Fixed (\\$000s)",
                                                                           "RdYlGn",  True,  1e3),
]

W_LABELS_SHORT  = [str(w) for w in WS]
G_LABELS_SHORT  = [str(g) for g in GAMMAS]
panel_labels    = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for ax, (col, title, cmap, diverge, scale), lbl in \
        zip(axes.flatten(), metrics, panel_labels):

    piv = sub075.pivot(index="w", columns="gamma", values=col).astype(float)
    vals = piv.values / scale

    # Mask degenerate corner cases (rob_profit_LB ≈ 0)
    mask = sub075.pivot(index="w", columns="gamma",
                        values="rob_profit_LB").values < 1.0
    vals[mask] = np.nan

    if diverge:
        vabs = np.nanmax(np.abs(vals))
        im = ax.imshow(vals, cmap=cmap, vmin=-vabs, vmax=vabs,
                       aspect="auto", origin="lower")
    else:
        im = ax.imshow(vals, cmap=cmap, aspect="auto", origin="lower")

    plt.colorbar(im, ax=ax, shrink=0.85)

    ax.set_xticks(range(len(GAMMAS)))
    ax.set_xticklabels([f"$\\Gamma={g}$" for g in GAMMAS], fontsize=8)
    ax.set_yticks(range(len(WS)))
    ax.set_yticklabels([f"$w={w}$" for w in WS], fontsize=8)
    ax.set_xlabel("Uncertainty budget", fontsize=9)
    ax.set_ylabel("Congestion cost", fontsize=9)
    ax.set_title(f"{lbl} {title}", loc="left", fontweight="bold", fontsize=9)

    for i in range(len(WS)):
        for j in range(len(GAMMAS)):
            v = vals[i, j]
            if np.isnan(v):
                txt = "—"
            elif scale == 1.0:
                txt = f"{v:.1f}%"
            elif abs(v) >= 100:
                txt = f"{v:.0f}"
            else:
                txt = f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if (not np.isnan(v) and
                                      abs(v) > 0.6 * np.nanmax(np.abs(vals)))
                    else "black")

fig4.suptitle("Sensitivity Analysis  ($v = 0.75$)",
              fontsize=11, fontweight="bold", y=1.01)
fig4.tight_layout(pad=1.5)
save(fig4, "fig4_sensitivity_heatmaps")
plt.close(fig4)

print("\nAll figures generated successfully.")
