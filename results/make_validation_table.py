#!/usr/bin/env python3
"""Per-FM precision, recall, F1, and Cohen's kappa table from human_validation.csv."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
MAST_ORDER = list(FM_NAMES.keys())

df = pd.read_csv("human_validation.csv")
df = df[df["human"] != "excluded"].copy()
df["human"] = pd.to_numeric(df["human"], errors="coerce")
df["judge"] = pd.to_numeric(df["judge"], errors="coerce")
df = df.dropna(subset=["human", "judge"])
df["human"] = df["human"].astype(int)
df["judge"] = df["judge"].astype(int)
# mode column is float64 (1.1, 1.2 …) — normalise to string for matching
df["mode"] = df["mode"].astype(float).apply(lambda x: f"{x:.1f}")


def cohen_kappa(h, j):
    n = len(h)
    if n == 0:
        return np.nan
    h, j = np.array(h), np.array(j)
    po = (h == j).mean()
    p_yes_h, p_yes_j = h.mean(), j.mean()
    pe = p_yes_h * p_yes_j + (1 - p_yes_h) * (1 - p_yes_j)
    return np.nan if pe == 1.0 else (po - pe) / (1 - pe)


rows = []
for fm in MAST_ORDER:
    sub = df[df["mode"] == fm]
    h, j = sub["human"].values, sub["judge"].values
    n = len(h)
    tp = int(((h == 1) & (j == 1)).sum())
    fp = int(((h == 0) & (j == 1)).sum())
    fn = int(((h == 1) & (j == 0)).sum())
    tn = int(((h == 0) & (j == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    kap  = cohen_kappa(h, j)
    rows.append(dict(fm=fm, name=FM_NAMES[fm], n=n,
                     tp=tp, fp=fp, fn=fn, tn=tn,
                     precision=prec, recall=rec, f1=f1, kappa=kap))

stats = pd.DataFrame(rows)

total_tp = int(stats["tp"].sum())
total_fp = int(stats["fp"].sum())
total_fn = int(stats["fn"].sum())
micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
              if (micro_prec + micro_rec) > 0 else 0.0)
overall_kappa  = cohen_kappa(df["human"].values, df["judge"].values)
overall_agree  = (df["human"] == df["judge"]).mean()

# Quick sanity print
for _, r in stats.iterrows():
    print(f"{r['fm']}  n={r['n']:3d}  TP={r['tp']:3d}  FP={r['fp']:3d}"
          f"  FN={r['fn']:3d}  TN={r['tn']:3d}"
          f"  P={r['precision']:.2f}  R={r['recall']:.2f}"
          f"  F1={r['f1']:.2f}  κ={r['kappa']:.3f}" if not np.isnan(r['kappa'])
          else f"{r['fm']}  n={r['n']:3d}  TP={r['tp']:3d}  FP={r['fp']:3d}"
               f"  FN={r['fn']:3d}  TN={r['tn']:3d}"
               f"  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}  κ=—")
print(f"\nMICRO  TP={total_tp} FP={total_fp} FN={total_fn}"
      f"  P={micro_prec:.3f}  R={micro_rec:.3f}  F1={micro_f1:.3f}  κ={overall_kappa:.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
N = len(stats)

# Column spec: (header, x_left, width, text_align)
COLS = [
    ("FM",         0.000, 0.058, "center"),
    ("Failure Mode", 0.060, 0.230, "left"),
    ("n",          0.292, 0.038, "center"),
    ("TP",         0.332, 0.045, "center"),
    ("FP",         0.379, 0.045, "center"),
    ("FN",         0.426, 0.045, "center"),
    ("TN",         0.473, 0.045, "center"),
    ("Precision",  0.524, 0.115, "center"),
    ("Recall",     0.641, 0.115, "center"),
    ("F1",         0.758, 0.100, "center"),
    ("Cohen's κ",  0.862, 0.130, "center"),
]

ROW_H   = 0.50
HDR_H   = 0.65
TITLE_H = 0.70
FOOT_H  = 0.55
PAD_B   = 0.30
fig_h = TITLE_H + HDR_H + N * ROW_H + FOOT_H + PAD_B

fig, ax = plt.subplots(figsize=(15, fig_h))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)

# y-coordinate helpers (working top-to-bottom in data space)
title_top  = fig_h - 0.08
hdr_top    = title_top - TITLE_H
rows_top   = hdr_top - HDR_H
foot_top   = rows_top - N * ROW_H


def hdr_cy():
    return hdr_top - HDR_H / 2

def row_cy(i):
    return rows_top - i * ROW_H - ROW_H / 2

def row_bot(i):
    return rows_top - i * ROW_H - ROW_H

def foot_cy():
    return foot_top - FOOT_H / 2


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, title_top - 0.06,
        "Human vs LLM Judge — Per-FM Precision, Recall, F1 & Cohen's κ",
        transform=ax.transData, ha="center", va="top",
        fontsize=14, fontweight="bold", color="#e0e8ff")
ax.text(0.5, title_top - 0.36,
        "LLM judge as predictor · human annotation as ground truth  ·  "
        f"Overall agreement {overall_agree:.1%}  ·  "
        f"n={len(df)} (19 traces × 2 models, excluded rows removed)",
        transform=ax.transData, ha="center", va="top",
        fontsize=9, color="#777")

# ── Header row ────────────────────────────────────────────────────────────────
for label, xs, w, align in COLS:
    rect = plt.Rectangle(
        (xs + 0.003, hdr_top - HDR_H + 0.04), w - 0.006, HDR_H - 0.06,
        fc="#1a3a5c", lw=0, transform=ax.transData, clip_on=False,
    )
    ax.add_patch(rect)
    tx = xs + w / 2 if align == "center" else xs + 0.008
    ax.text(tx, hdr_cy(), label,
            transform=ax.transData, ha=align, va="center",
            fontsize=10, fontweight="bold", color="#b0c8f0",
            multialignment="center")

# ── Color helpers ─────────────────────────────────────────────────────────────
def metric_color(v, thresholds=(0.60, 0.35)):
    if np.isnan(v): return "#666"
    if v >= thresholds[0]: return "#2ecc71"
    if v >= thresholds[1]: return "#e67e22"
    return "#e74c3c"

def kappa_color(v):
    if np.isnan(v): return "#666"
    if v >= 0.40: return "#2ecc71"
    if v >= 0.20: return "#e67e22"
    if v >= 0.00: return "#e8a020"
    return "#e74c3c"

def bar_underlay(xs, w, cy, rh, val, color, alpha=0.22):
    if np.isnan(val) or val <= 0:
        return
    fill_w = (w - 0.008) * min(1.0, max(0.0, val))
    rect = plt.Rectangle(
        (xs + 0.004, cy - rh / 2 + 0.04), fill_w, rh - 0.08,
        fc=color, alpha=alpha, lw=0,
        transform=ax.transData, clip_on=False,
    )
    ax.add_patch(rect)

# ── Data rows ─────────────────────────────────────────────────────────────────
for i, (_, row) in enumerate(stats.iterrows()):
    cy  = row_cy(i)
    rh  = ROW_H
    bot = row_bot(i)
    row_bg = "#161b22" if i % 2 == 0 else "#0d1117"

    bg = plt.Rectangle((0.001, bot + 0.03), 0.997, rh - 0.04,
                        fc=row_bg, lw=0, transform=ax.transData, clip_on=False)
    ax.add_patch(bg)

    prec = row["precision"]
    rec  = row["recall"]
    f1   = row["f1"]
    kap  = row["kappa"]

    # bar underlays for metric columns
    for xs, w, val, col_fn in [
        (0.524, 0.115, prec, metric_color),
        (0.641, 0.115, rec,  metric_color),
        (0.758, 0.100, f1,   metric_color),
        (0.862, 0.130, max(0, kap) if not np.isnan(kap) else np.nan, kappa_color),
    ]:
        bar_underlay(xs, w, cy, rh, val, col_fn(val))

    cells = [
        (row["fm"],       0.000, 0.058, "center", "#7ab8f5", 9.5, True),
        (row["name"],     0.060, 0.230, "left",   "#ccc",    9,   False),
        (str(row["n"]),   0.292, 0.038, "center", "#888",    9,   False),
        (str(row["tp"]),  0.332, 0.045, "center", "#5dade2", 9,   False),
        (str(row["fp"]),  0.379, 0.045, "center", "#e87070", 9,   False),
        (str(row["fn"]),  0.426, 0.045, "center", "#e8a870", 9,   False),
        (str(row["tn"]),  0.473, 0.045, "center", "#82e0aa", 9,   False),
        (f"{prec:.2f}",   0.524, 0.115, "center", metric_color(prec), 10, True),
        (f"{rec:.2f}",    0.641, 0.115, "center", metric_color(rec),  10, True),
        (f"{f1:.2f}",     0.758, 0.100, "center", metric_color(f1),   10, True),
        (f"{kap:.3f}" if not np.isnan(kap) else "—",
                          0.862, 0.130, "center", kappa_color(kap), 10, True),
    ]
    for text, xs, w, align, color, fs, bold in cells:
        tx = xs + w / 2 if align == "center" else xs + 0.010
        ax.text(tx, cy, text,
                transform=ax.transData, ha=align, va="center",
                fontsize=fs, color=color,
                fontweight="bold" if bold else "normal")

# ── Footer: micro overall ─────────────────────────────────────────────────────
bg = plt.Rectangle((0.001, foot_top - FOOT_H + 0.03), 0.997, FOOT_H - 0.04,
                    fc="#0d2a1f", lw=0, transform=ax.transData, clip_on=False)
ax.add_patch(bg)

fcy = foot_cy()
for label, xs, w, color, fs, bold in [
    ("MICRO OVERALL",         0.060, 0.230, "#5dde90", 10, True),
    (f"n={len(df)}",          0.292, 0.083, "#999",    9,  False),
    (f"TP={total_tp}",        0.332, 0.045, "#5dade2", 9,  False),
    (f"FP={total_fp}",        0.379, 0.045, "#e87070", 9,  False),
    (f"FN={total_fn}",        0.426, 0.045, "#e8a870", 9,  False),
    (f"{micro_prec:.3f}",     0.524, 0.115, metric_color(micro_prec), 11, True),
    (f"{micro_rec:.3f}",      0.641, 0.115, metric_color(micro_rec),  11, True),
    (f"{micro_f1:.3f}",       0.758, 0.100, metric_color(micro_f1),   11, True),
    (f"{overall_kappa:.3f}",  0.862, 0.130, kappa_color(overall_kappa), 11, True),
]:
    tx = xs + w / 2 if label not in ("MICRO OVERALL",) else xs + 0.010
    ax.text(tx, fcy, label,
            transform=ax.transData, ha="left" if label == "MICRO OVERALL" else "center",
            va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal")

# ── Legend ────────────────────────────────────────────────────────────────────
leg_y = foot_top - FOOT_H - 0.12
ax.text(0.524, leg_y, "Color threshold — Precision / Recall / F1 / κ:",
        transform=ax.transData, ha="left", va="center", fontsize=8, color="#666")
lx = 0.524
for color, label in [("#2ecc71","≥ 0.60"), ("#e67e22","0.35–0.59"), ("#e74c3c","< 0.35")]:
    p = plt.Rectangle((lx + 0.155, leg_y - 0.07), 0.020, 0.13,
                       fc=color, alpha=0.5, lw=0,
                       transform=ax.transData, clip_on=False)
    ax.add_patch(p)
    ax.text(lx + 0.178, leg_y, label,
            transform=ax.transData, ha="left", va="center", fontsize=8, color="#999")
    lx += 0.10

out = "validation_table.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out}")
