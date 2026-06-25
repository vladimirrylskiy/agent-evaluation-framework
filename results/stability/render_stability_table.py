#!/usr/bin/env python3
"""Render stochastic stability results as a publication-quality PNG table."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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

CSV_PATH = Path(__file__).parent / "stability_summary_metagpt_6_20260619_111723.csv"
OUT_PATH = Path(__file__).parent / "stability_metagpt_n30.png"


def pct_cell(agreement, n, is_judge=False):
    """Format a cell value and return (text, color)."""
    if not is_judge and (n == 0 or n == "" or n is None):
        return "—", "#2a2a2a"
    try:
        val = float(agreement)
    except (ValueError, TypeError):
        return "—", "#2a2a2a"
    pct = int(round(val * 100))
    if is_judge:
        label = f"{pct}%"
    else:
        label = f"{pct}%\n(n={n})"
    # Color: green ≥ 80%, yellow 60–79%, red < 60%
    if val >= 0.80:
        color = "#1a472a"   # dark green
    elif val >= 0.60:
        color = "#4a3800"   # dark amber
    else:
        color = "#4a0f0f"   # dark red
    return label, color


def text_color(bg):
    if bg == "#2a2a2a":
        return "#666666"
    return "#ffffff"


def main():
    rows_data = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_data.append(row)

    col_headers = ["FM", "Failure Mode", "Judge\n(N=30)", "Forced", "Semi-Forced", "Relaxed"]
    col_widths   = [0.07, 0.30, 0.13, 0.16, 0.16, 0.16]

    n_rows = len(rows_data)
    n_cols = len(col_headers)

    fig_w = 14
    row_h = 0.52
    header_h = 0.7
    fig_h = header_h + n_rows * row_h + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")
    ax.axis("off")

    # Absolute column x positions
    x_starts = []
    x = 0.01
    for w in col_widths:
        x_starts.append(x)
        x += w

    total_w = sum(col_widths)
    y_top = 1.0 - (header_h / fig_h)

    def row_y(i):
        return y_top - (i + 1) * (row_h / fig_h)

    # ── Header ────────────────────────────────────────────────────────────────
    for ci, (hdr, xs, w) in enumerate(zip(col_headers, x_starts, col_widths)):
        rect = mpatches.FancyBboxPatch(
            (xs + 0.002, y_top + 0.005), w - 0.005, header_h / fig_h - 0.01,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor="#1e3a5f", transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            xs + w / 2, y_top + (header_h / fig_h) / 2,
            hdr, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#e0e8ff",
            multialignment="center",
        )

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, row in enumerate(rows_data):
        fm   = row["fm"]
        name = FM_NAMES.get(fm, fm)
        ry   = row_y(ri)
        row_bg = "#1a1a1a" if ri % 2 == 0 else "#141414"

        # Background stripe
        stripe = mpatches.FancyBboxPatch(
            (0.01, ry + 0.003), total_w - 0.002, row_h / fig_h - 0.006,
            boxstyle="round,pad=0.002", linewidth=0,
            facecolor=row_bg, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(stripe)

        cells = []

        # FM code
        cells.append((fm, "#1e3a5f", "#a8c4ff"))

        # FM name — highlight if judge majority = 1 (present)
        majority = row.get("judge_majority", "0")
        name_bg  = "#1a2a1a" if str(majority) == "1" else row_bg
        name_col = "#90ee90" if str(majority) == "1" else "#cccccc"
        cells.append((name, name_bg, name_col))

        # Judge
        jlabel, jbg = pct_cell(row["judge_agreement"], 30, is_judge=True)
        cells.append((jlabel, jbg, text_color(jbg)))

        # Forced
        flabel, fbg = pct_cell(row["forced_agreement"], row["forced_n"])
        cells.append((flabel, fbg, text_color(fbg)))

        # Semi-Forced
        slabel, sbg = pct_cell(row["semi_agreement"], row["semi_n"])
        cells.append((slabel, sbg, text_color(sbg)))

        # Relaxed
        rlabel, rbg = pct_cell(row["relaxed_agreement"], 30, is_judge=True)
        cells.append((rlabel, rbg, text_color(rbg)))

        for ci, ((text, bg, fg), xs, w) in enumerate(zip(cells, x_starts, col_widths)):
            if bg not in (row_bg, "#1a2a1a"):
                cell_rect = mpatches.FancyBboxPatch(
                    (xs + 0.003, ry + 0.005), w - 0.007, row_h / fig_h - 0.01,
                    boxstyle="round,pad=0.003", linewidth=0,
                    facecolor=bg, transform=ax.transAxes, clip_on=False
                )
                ax.add_patch(cell_rect)
            elif bg == "#1a2a1a":
                cell_rect = mpatches.FancyBboxPatch(
                    (xs + 0.003, ry + 0.005), w - 0.007, row_h / fig_h - 0.01,
                    boxstyle="round,pad=0.003", linewidth=0,
                    facecolor=bg, transform=ax.transAxes, clip_on=False
                )
                ax.add_patch(cell_rect)

            ax.text(
                xs + w / 2, ry + (row_h / fig_h) / 2,
                text, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9 if ci == 1 else 9.5,
                color=fg,
                multialignment="center",
                fontweight="bold" if ci == 0 else "normal",
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = row_y(n_rows) - 0.01
    legend_items = [
        ("#1a472a", "≥ 80% agreement"),
        ("#4a3800", "60–79% agreement"),
        ("#4a0f0f", "< 60% agreement"),
        ("#1a2a1a", "FM detected (judge majority = present)"),
    ]
    lx = 0.01
    for bg, label in legend_items:
        patch = mpatches.FancyBboxPatch(
            (lx, legend_y - 0.005), 0.018, 0.022,
            boxstyle="round,pad=0.002", linewidth=0,
            facecolor=bg, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(patch)
        ax.text(lx + 0.022, legend_y + 0.006, label,
                transform=ax.transAxes, fontsize=8, color="#999999", va="center")
        lx += 0.19

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(
        0.01 + total_w / 2,
        y_top + header_h / fig_h + 0.04,
        "Stochastic Stability — MetaGPT Trace 6 (N = 30 runs, temperature = 1.0)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12, fontweight="bold", color="#e0e8ff",
    )
    ax.text(
        0.01 + total_w / 2,
        y_top + header_h / fig_h + 0.015,
        "Agreement = fraction of runs matching the majority verdict  ·  n = runs where localizer was triggered (judge said present)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color="#888888",
    )

    plt.tight_layout(pad=0)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
