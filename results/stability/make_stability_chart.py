#!/usr/bin/env python3
"""Nice chart for the stochastic stability test of step localisation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe

FM_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No or Incomplete Verification",
    "3.3": "Incorrect Verification",
}

df = pd.read_csv("stability_summary_metagpt_6_20260619_111723.csv")

# ── Build matrices ──────────────────────────────────────────────────────────────
conditions = [
    ("Judge",          "judge_agreement",  "judge_n"),
    ("Forced",         "forced_agreement", "forced_n"),
    ("Semi-Forced",    "semi_agreement",   "semi_n"),
    ("Relaxed",        "relaxed_agreement","relaxed_n"),
]

fms = df["fm"].astype(str).tolist()
N = len(fms)
C = len(conditions)

heat   = np.full((N, C), np.nan)
annot  = [[""] * C for _ in range(N)]
ns_mat = np.zeros((N, C))

for ci, (label, agree_col, n_col) in enumerate(conditions):
    for ri, row in df.iterrows():
        try:
            n = int(float(row[n_col]))
        except (ValueError, TypeError):
            n = 0
        try:
            a = float(row[agree_col])
        except (ValueError, TypeError):
            a = np.nan

        ns_mat[ri, ci] = n
        is_always_n30 = (ci == 0) or (ci == 3)          # judge & relaxed always run

        if is_always_n30:
            if not np.isnan(a):
                heat[ri, ci] = a
                if ci == 0:
                    majority = df.iloc[ri]["judge_majority"]
                    verdict = "\n(present)" if majority == 1 else "\n(absent)"
                    annot[ri][ci] = f"{int(round(a*100))}%{verdict}"
                else:
                    annot[ri][ci] = f"{int(round(a*100))}%"
        else:
            if n > 0 and not np.isnan(a):
                heat[ri, ci] = a
                annot[ri][ci] = f"{int(round(a*100))}%\n(n={n})"

# ── Color map  ──────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "rag",
    [(0.0, "#c0392b"), (0.4, "#e67e22"), (0.6, "#f1c40f"),
     (0.80, "#27ae60"), (1.0, "#1a5e20")],
)
cmap.set_bad("#dce3ec")       # grey-blue for N/A
norm = Normalize(vmin=0, vmax=1)

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8.5))
plt.subplots_adjust(left=0.30, right=0.87, top=0.82, bottom=0.10)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Draw heatmap manually so we can customise each cell
cell_h = 1.0
cell_w = 1.0

for ri in range(N):
    for ci in range(C):
        val = heat[ri, ci]
        color = cmap(norm(val)) if not np.isnan(val) else cmap(np.nan)
        rect = plt.Rectangle((ci, N - ri - 1), cell_w, cell_h,
                               color=color, ec="white", lw=1.5)
        ax.add_patch(rect)

        txt = annot[ri][ci]
        if txt:
            lum = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            fg = "white" if lum < 0.55 else "#222"
            ax.text(ci + 0.5, N - ri - 0.5, txt,
                    ha="center", va="center", fontsize=9.5,
                    color=fg, multialignment="center", fontweight="bold")
        elif not np.isnan(val):
            pass
        else:
            ax.text(ci + 0.5, N - ri - 0.5, "—",
                    ha="center", va="center", fontsize=11,
                    color="#9baab8")

# ── FM row labels ──────────────────────────────────────────────────────────────
for ri, fm in enumerate(fms):
    y = N - ri - 0.5
    is_detected = df.iloc[ri]["judge_majority"] == 1

    # coloured FM code chip
    chip_bg = "#2471a3" if is_detected else "#7f8c8d"
    ax.text(-0.12, y, fm,
            ha="right", va="center", fontsize=8.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=chip_bg, ec="none"))

    name_color = "#1a252f" if is_detected else "#5d6d7e"
    name_weight = "semibold" if is_detected else "normal"
    ax.text(-0.22, y, FM_NAMES[fm],
            ha="right", va="center", fontsize=8.5,
            color=name_color, fontweight=name_weight)

# Green outline for detected FMs
for ri, fm in enumerate(fms):
    if df.iloc[ri]["judge_majority"] == 1:
        box = plt.Rectangle((0, N - ri - 1), C, 1,
                             fill=False, ec="#27ae60", lw=2, zorder=5)
        ax.add_patch(box)

# ── Column headers ─────────────────────────────────────────────────────────────
col_labels = ["Judge\n(N=30)", "Forced\nLocalizer", "Semi-Forced\nLocalizer", "Relaxed\nLocalizer"]
for ci, label in enumerate(col_labels):
    ax.text(ci + 0.5, N + 0.15, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#1a252f")

# Section separators between judge / localizers
ax.axvline(1, color="#7f8c8d", lw=1.5, linestyle="--", alpha=0.5)

ax.set_xlim(0, C)
ax.set_ylim(0, N)
ax.axis("off")

# ── Colorbar ───────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.89, 0.12, 0.018, 0.65])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Agreement rate", fontsize=9, color="#333")
cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(["0%","20%","40%","60%","80%","100%"])
cb.ax.tick_params(labelsize=8, color="#555")
cb.outline.set_edgecolor("#aaa")

# N/A patch in legend
na_patch = mpatches.Patch(facecolor="#dce3ec", edgecolor="white", label="Localizer not triggered")
detected_patch = mpatches.Patch(facecolor="none", edgecolor="#27ae60",
                                 linewidth=2, label="FM detected (judge majority = present)")
fig.legend(handles=[na_patch, detected_patch],
           loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=2, fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
           bbox_transform=fig.transFigure)

# ── Title ──────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "Stochastic Stability — MetaGPT Trace 6  (N = 30 runs, temperature = 1.0)",
         ha="center", va="center", fontsize=13, fontweight="bold", color="#1a252f",
         transform=fig.transFigure)
fig.text(0.5, 0.915,
         "Agreement = fraction of runs matching majority verdict  ·  "
         "n = triggered runs  ·  dashed line separates detection (left) from localization (right)",
         ha="center", va="center", fontsize=8, color="#666",
         transform=fig.transFigure)

out = "stability_chart.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
