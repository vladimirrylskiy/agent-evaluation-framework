#!/usr/bin/env python3
"""
FM step-position distribution across the trace.
Shows WHERE in the trace each failure mode tends to occur (pct = 0–100 % through trace).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ── Data ─────────────────────────────────────────────────────────────────────
raw = pd.read_csv("fm_scan_raw_20260618_233820.csv")

FM_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conv. History",
    "1.5": "Unaware of Termination",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Agent Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No / Incomplete Verification",
    "3.3": "Incorrect Verification",
}
MAST_ORDER = ["1.1","1.2","1.3","1.4","1.5",
               "2.1","2.2","2.3","2.4","2.5","2.6",
               "3.1","3.2","3.3"]

FW_COLORS = {
    "ChatDev":     "#e74c3c",
    "AG2":         "#3498db",
    "AppWorld":    "#2ecc71",
    "HyperAgent":  "#f39c12",
    "MetaGPT":     "#9b59b6",
    "MagenticOne": "#16a085",
    "OpenManus":   "#e67e22",
}

raw["fm"] = raw["fm"].astype(str)
detected = raw[raw["present"] == 1].copy()

# Step detections (have a pct value)
step_det = detected[detected["is_global"] == 0].dropna(subset=["pct"]).copy()
step_det["pct"] = step_det["pct"].astype(float)

# Global detections (span whole trace — no step position)
global_det = detected[detected["is_global"] == 1].copy()

# ── Figure layout ─────────────────────────────────────────────────────────────
N = len(MAST_ORDER)
ROW_H = 1.0

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")

gs = GridSpec(
    1, 2, figure=fig,
    left=0.34, right=0.97, top=0.87, bottom=0.10,
    wspace=0.03, width_ratios=[14, 1],
)
ax  = fig.add_subplot(gs[0])   # main strip plot
axn = fig.add_subplot(gs[1])   # count bar

for a in (ax, axn):
    a.set_facecolor("white")
    a.spines[:].set_visible(False)

# ── Row backgrounds & separators ──────────────────────────────────────────────
for ri, fm in enumerate(MAST_ORDER):
    y0 = (N - ri - 1) * ROW_H
    bg = "#f8f9fa" if ri % 2 == 0 else "white"
    ax.axhspan(y0, y0 + ROW_H, color=bg, zorder=0)
    # category separator
    cat = fm.split(".")[0]
    next_cat = MAST_ORDER[ri + 1].split(".")[0] if ri + 1 < N else None
    if next_cat and next_cat != cat:
        ax.axhline(y0, color="#bdc3c7", lw=1.2, zorder=1)

# ── KDE density ribbons ────────────────────────────────────────────────────────
kde_x = np.linspace(0, 100, 300)
for ri, fm in enumerate(MAST_ORDER):
    y_center = (N - ri - 1) * ROW_H + ROW_H / 2
    pts = step_det[step_det["fm"] == fm]["pct"].values
    if len(pts) < 8:
        continue
    try:
        kde = gaussian_kde(pts, bw_method=0.25)
        density = kde(kde_x)
        density = density / density.max() * 0.38   # scale to ±0.38 of row height
        ax.fill_between(kde_x, y_center - density, y_center + density,
                        color="#2c3e50", alpha=0.10, zorder=1)
        ax.plot(kde_x, y_center + density, color="#7f8c8d", lw=0.6, zorder=2)
        ax.plot(kde_x, y_center - density, color="#7f8c8d", lw=0.6, zorder=2)
    except Exception:
        pass

# ── Strip dots ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

for ri, fm in enumerate(MAST_ORDER):
    y_center = (N - ri - 1) * ROW_H + ROW_H / 2
    fm_pts = step_det[step_det["fm"] == fm]

    for _, row in fm_pts.iterrows():
        jitter = rng.uniform(-0.28, 0.28)
        color  = FW_COLORS.get(row["framework"], "#95a5a6")
        ax.scatter(
            row["pct"], y_center + jitter,
            s=22, color=color, alpha=0.75,
            edgecolors="none", zorder=4, linewidths=0,
        )

# ── FM labels ─────────────────────────────────────────────────────────────────
for ri, fm in enumerate(MAST_ORDER):
    y_center = (N - ri - 1) * ROW_H + ROW_H / 2

    # FM number chip — sits just left of the plot area
    ax.text(-1.0, y_center, fm,
            ha="right", va="center", fontsize=10, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.30", fc="#2471a3", ec="none"),
            transform=ax.transData)
    # FM name — clear gap (≈3 units) to the left of the chip
    ax.text(-5.5, y_center, FM_NAMES[fm],
            ha="right", va="center", fontsize=10, color="#1a252f",
            transform=ax.transData)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-2, 104)
ax.set_ylim(0, N * ROW_H)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%\n(start)", "25%", "50%\n(mid)", "75%", "100%\n(end)"],
                   fontsize=9, color="#555")
ax.tick_params(axis="x", length=4, color="#bbb", bottom=True)
ax.tick_params(axis="y", left=False)
ax.set_yticks([])
ax.xaxis.set_label_position("bottom")
ax.set_xlabel("Position in trace (% through)", fontsize=10, color="#333", labelpad=8)

# Vertical reference lines
for xv in [25, 50, 75]:
    ax.axvline(xv, color="#e0e0e0", lw=0.8, linestyle="--", zorder=0)

# ── Count bars (right panel) ──────────────────────────────────────────────────
max_n = max(len(step_det[step_det["fm"] == fm]) for fm in MAST_ORDER)
for ri, fm in enumerate(MAST_ORDER):
    y0 = (N - ri - 1) * ROW_H + 0.18
    n_step   = len(step_det[step_det["fm"] == fm])
    n_global = len(global_det[global_det["fm"] == fm])

    if n_step > 0:
        axn.barh(y0, n_step, height=0.64, left=0,
                 color="#2980b9", align="edge", alpha=0.85)
    if n_global > 0:
        axn.barh(y0, n_global, height=0.64, left=n_step,
                 color="#e74c3c", align="edge", alpha=0.7)

    total = n_step + n_global
    axn.text(max_n * 0.55, y0 + 0.32, str(total) if total else "—",
             ha="center", va="center", fontsize=7.5,
             color="white" if total > max_n * 0.3 else "#555",
             fontweight="bold")

axn.set_xlim(0, max_n * 1.05)
axn.set_ylim(0, N * ROW_H)
axn.axis("off")
axn.text(max_n * 0.5, N * ROW_H + 0.15, "n",
         ha="center", va="bottom", fontsize=9, color="#555", fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
fw_patches = [mpatches.Patch(fc=c, ec="none", label=fw)
              for fw, c in FW_COLORS.items()]
global_patch = mpatches.Patch(fc="#e74c3c", ec="none", alpha=0.7,
                               label="Global detection (counted in n only)")
kde_patch = mpatches.Patch(fc="#2c3e50", ec="none", alpha=0.15,
                            label="Density (KDE, n ≥ 8)")

fig.legend(
    handles=fw_patches + [global_patch, kde_patch],
    loc="lower center", bbox_to_anchor=(0.55, 0.005),
    ncol=5, fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
    bbox_transform=fig.transFigure,
)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.57, 0.955,
         "Failure Mode Distribution Across Trace Position",
         ha="center", va="center", fontsize=13, fontweight="bold", color="#1a252f",
         transform=fig.transFigure)
fig.text(0.57, 0.923,
         "Each dot = one detected step occurrence · position = % through the trace · "
         "shaded ribbon = KDE density",
         ha="center", va="center", fontsize=8.5, color="#666",
         transform=fig.transFigure)

out = "fm_trace_distribution.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
