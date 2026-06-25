#!/usr/bin/env python3
"""
Distribution + stability chart for the stochastic step-localisation experiment.
Shows, per failure mode, which of the 30 runs detected it and how stable that is.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

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

raw     = pd.read_csv("stability_raw_metagpt_6_20260619_111723.csv")
summary = pd.read_csv("stability_summary_metagpt_6_20260619_111723.csv")

fms = [str(f) for f in summary["fm"].tolist()]   # preserve MAST order

# Per-FM detection count
det_count = {str(r["fm"]): int(raw[raw["fm"] == r["fm"]]["judge"].sum())
             for _, r in summary.iterrows()}
agreement  = {str(r["fm"]): float(r["judge_agreement"]) for _, r in summary.iterrows()}
majority   = {str(r["fm"]): int(r["judge_majority"])    for _, r in summary.iterrows()}

# ── Colour helpers ────────────────────────────────────────────────────────────
DETECTED_COLOR = "#2980b9"
ABSENT_COLOR   = "#dce3ec"
MAJORITY_MATCH = "#2980b9"   # run agrees with majority
MINORITY_COLOR = "#e74c3c"   # run disagrees with majority

agree_cmap = LinearSegmentedColormap.from_list(
    "agree", [(0.0, "#e74c3c"), (0.5, "#e67e22"), (0.8, "#27ae60"), (1.0, "#145a32")]
)

# ── Layout ────────────────────────────────────────────────────────────────────
N_RUNS = 30
N_FM   = len(fms)

fig = plt.figure(figsize=(16, 8.5))
fig.patch.set_facecolor("white")

# GridSpec: [run grid | detection bar | agreement bar]
gs = GridSpec(
    1, 3,
    figure=fig,
    left=0.24, right=0.97,
    top=0.87,  bottom=0.08,
    wspace=0.04,
    width_ratios=[30, 5, 3],
)

ax_grid  = fig.add_subplot(gs[0])   # 30-run dot grid
ax_det   = fig.add_subplot(gs[1])   # detection count bar
ax_agree = fig.add_subplot(gs[2])   # agreement rate

for ax in (ax_grid, ax_det, ax_agree):
    ax.set_facecolor("white")
    ax.spines[:].set_visible(False)

# ── Run grid ──────────────────────────────────────────────────────────────────
row_h = 1.0
col_w = 1.0
pad   = 0.12

for ri, fm in enumerate(fms):
    y_center = (N_FM - ri - 1) * row_h + row_h / 2
    maj = majority[fm]

    for run_i in range(1, N_RUNS + 1):
        x_center = (run_i - 1) * col_w + col_w / 2

        # get judge value for this run & FM
        cell = raw[(raw["run"] == run_i) & (raw["fm"].astype(str) == fm)]
        judge_val = int(cell["judge"].iloc[0]) if len(cell) else 0

        detected = judge_val == 1
        matches_majority = (judge_val == maj)

        # outer circle (always drawn)
        circle_bg = plt.Circle(
            (x_center, y_center), (col_w / 2 - pad),
            color=ABSENT_COLOR, zorder=2
        )
        ax_grid.add_patch(circle_bg)

        if detected:
            # filled circle, color by whether it agrees with majority
            color = MAJORITY_MATCH if matches_majority else MINORITY_COLOR
            dot = plt.Circle(
                (x_center, y_center), (col_w / 2 - pad),
                color=color, zorder=3
            )
            ax_grid.add_patch(dot)
        elif not matches_majority:
            # absent but majority says present → minority dissent, red ring
            ring = plt.Circle(
                (x_center, y_center), (col_w / 2 - pad),
                color=MINORITY_COLOR, zorder=3
            )
            ax_grid.add_patch(ring)

# FM labels (left of grid)
category_sep = {"1": "Category 1\nAgent–Task", "2": "Category 2\nAgent–Agent", "3": "Category 3\nOutput"}
prev_cat = None
for ri, fm in enumerate(fms):
    y_center = (N_FM - ri - 1) * row_h + row_h / 2
    cat = fm.split(".")[0]

    # Category divider line
    if cat != prev_cat and ri > 0:
        ax_grid.axhline(y=(N_FM - ri) * row_h, color="#bdc3c7", lw=1.2, xmin=-10,
                        clip_on=False, zorder=1)
    prev_cat = cat

    chip_bg = "#2471a3" if majority[fm] == 1 else "#7f8c8d"
    ax_grid.text(-0.6, y_center, fm,
                 ha="right", va="center", fontsize=8, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.22", fc=chip_bg, ec="none"),
                 transform=ax_grid.transData)
    ax_grid.text(-0.9, y_center, FM_NAMES[fm],
                 ha="right", va="center", fontsize=8.5,
                 color="#1a252f" if majority[fm] == 1 else "#5d6d7e",
                 fontweight="semibold" if majority[fm] == 1 else "normal")

# Run number labels on top
for run_i in range(1, N_RUNS + 1):
    x_center = (run_i - 1) * col_w + col_w / 2
    ax_grid.text(x_center, N_FM * row_h + 0.2, str(run_i),
                 ha="center", va="bottom", fontsize=6.5, color="#7f8c8d")

ax_grid.text(N_RUNS * col_w / 2, N_FM * row_h + 0.85,
             "Run #", ha="center", va="bottom", fontsize=8.5,
             color="#555", fontweight="bold")

ax_grid.set_xlim(0, N_RUNS * col_w)
ax_grid.set_ylim(0, N_FM * row_h)
ax_grid.axis("off")

# ── Detection count bars ──────────────────────────────────────────────────────
for ri, fm in enumerate(fms):
    y_bottom = (N_FM - ri - 1) * row_h + 0.18
    cnt = det_count[fm]

    # background bar (full 30 width)
    ax_det.barh(y_bottom, N_RUNS, height=0.64,
                left=0, color=ABSENT_COLOR, align="edge", zorder=1)
    # detected segment
    if cnt > 0:
        ax_det.barh(y_bottom, cnt, height=0.64,
                    left=0, color=DETECTED_COLOR, align="edge", zorder=2,
                    alpha=0.85)

    # annotation: "X / 30"
    ax_det.text(N_RUNS / 2, y_bottom + 0.32, f"{cnt}/30",
                ha="center", va="center", fontsize=8,
                color="white" if cnt >= 15 else "#2c3e50",
                fontweight="bold", zorder=3)

ax_det.set_xlim(0, N_RUNS)
ax_det.set_ylim(0, N_FM * row_h)
ax_det.set_xticks([0, 15, 30])
ax_det.set_xticklabels(["0", "15", "30"], fontsize=7.5, color="#555")
ax_det.xaxis.tick_top()
ax_det.tick_params(axis="x", length=3, color="#bbb")
ax_det.text(N_RUNS / 2, N_FM * row_h + 0.85, "Detected\n(# runs)",
            ha="center", va="bottom", fontsize=8.5, color="#555", fontweight="bold")
ax_det.axis("off")
ax_det.set_xlim(0, N_RUNS)
ax_det.set_ylim(0, N_FM * row_h)
for ri in range(N_FM + 1):
    ax_det.axhline(ri * row_h, color="white", lw=1.5, zorder=4)

# ── Agreement rate bars ───────────────────────────────────────────────────────
agree_norm = Normalize(vmin=0.5, vmax=1.0)

for ri, fm in enumerate(fms):
    y_bottom = (N_FM - ri - 1) * row_h + 0.18
    ag = agreement[fm]

    color = agree_cmap(agree_norm(ag))
    ax_agree.barh(y_bottom, ag, height=0.64,
                  left=0, color=color, align="edge", zorder=2)
    ax_agree.barh(y_bottom, 1 - ag, height=0.64,
                  left=ag, color="#f0f0f0", align="edge", zorder=1)

    lum = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
    fg = "white" if lum < 0.55 else "#222"
    ax_agree.text(ag / 2, y_bottom + 0.32, f"{int(round(ag*100))}%",
                  ha="center", va="center", fontsize=8,
                  color=fg, fontweight="bold", zorder=3)

ax_agree.set_xlim(0, 1)
ax_agree.set_ylim(0, N_FM * row_h)
ax_agree.axis("off")
ax_agree.text(0.5, N_FM * row_h + 0.85, "Judge\nAgreement",
              ha="center", va="bottom", fontsize=8.5, color="#555", fontweight="bold",
              transform=ax_agree.transData)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=DETECTED_COLOR,  ec="none", label="Detected in run (agrees with majority)"),
    mpatches.Patch(fc=MINORITY_COLOR,  ec="none", label="Minority verdict (disagrees with majority)"),
    mpatches.Patch(fc=ABSENT_COLOR,    ec="none", label="Not detected in run"),
]
fig.legend(handles=legend_items,
           loc="lower center", bbox_to_anchor=(0.6, 0.005),
           ncol=3, fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
           bbox_transform=fig.transFigure)

# ── Titles ────────────────────────────────────────────────────────────────────
fig.text(0.6, 0.955,
         "FM Detection Distribution & Stability — MetaGPT Trace 6  (N = 30 runs, temperature = 1.0)",
         ha="center", va="center", fontsize=12, fontweight="bold", color="#1a252f",
         transform=fig.transFigure)
fig.text(0.6, 0.922,
         "Each circle = one run  ·  Blue = FM detected  ·  Red = minority verdict  ·  "
         "Blue chip = FM present in majority of runs",
         ha="center", va="center", fontsize=8, color="#666",
         transform=fig.transFigure)

out = "stability_distribution_chart.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
