#!/usr/bin/env python3
"""Render the real-robot success-rate bar chart (Cube / Tomato / Average) for the site."""

import os
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import numpy as np

mpl.rcParams["font.family"] = "monospace"

# ── CONFIGURE YOUR VALUES HERE ──────────────────────────────────────────────

SUBPLOT_LABELS = ["Cube", "Tomato"]

DATA = {
    "Cube":   [90, 100, 90],
    "Tomato": [60, 70, 100],
}

STD = {
    "Cube":   [0, 0, 0],
    "Tomato": [0, 0, 0],
}

# ── APPEARANCE ───────────────────────────────────────────────────────────────

METHOD_LABELS = ["Base Pi0.5", "Teleop Finetuned", "Ctrl-World Finetuned"]

COLORS = ["#C46B53", "#AFC3C5", "#F9CF48"]
EDGE_COLORS = ["#000000", "#000000", "#000000"]

AVERAGE_BG       = "#ECECEC"
AVERAGE_BORDER   = "#9a9a9a"
GRID_COLOR       = "#d0d0d0"
SPINE_COLOR      = "#b0b0b0"
LEGEND_BORDER    = "#888888"
SHADOW_COLOR     = "#000000"
SHADOW_ALPHA     = 1.0
SHADOW_PX        = 3

HIGHLIGHT_IDX = len(METHOD_LABELS) - 1

# ── COMPUTE AVERAGE SUBPLOT ───────────────────────────────────────────────────

all_labels = SUBPLOT_LABELS + ["Average"]

avg_vals = [
    float(np.mean([DATA[sl][i] for sl in SUBPLOT_LABELS]))
    for i in range(len(METHOD_LABELS))
]
avg_stds = [
    float(np.sqrt(np.sum([STD[sl][i] ** 2 for sl in SUBPLOT_LABELS]))) / len(SUBPLOT_LABELS)
    for i in range(len(METHOD_LABELS))
]
all_data = {sl: DATA[sl] for sl in SUBPLOT_LABELS}
all_data["Average"] = avg_vals
all_std = {sl: STD[sl] for sl in SUBPLOT_LABELS}
all_std["Average"] = avg_stds

print("Average values:", avg_vals)

# ── PLOT ─────────────────────────────────────────────────────────────────────

n_methods = len(METHOD_LABELS)
bar_width = 0.11
spacing   = 0.14
offsets   = np.arange(n_methods) * spacing - (n_methods - 1) * spacing / 2

fig, axes = plt.subplots(1, 3, figsize=(8, 3.4), sharey=True)
fig.patch.set_facecolor("white")

for ax, sublabel in zip(axes, all_labels):
    vals = all_data[sublabel]
    stds = all_std[sublabel]

    if sublabel == "Average":
        ax.set_facecolor(AVERAGE_BG)
    else:
        ax.set_facecolor("white")

    for i, (offset, value, std, color, edge) in enumerate(
        zip(offsets, vals, stds, COLORS, EDGE_COLORS)
    ):
        lw = 0.9
        shadow_transform = ax.transData + transforms.ScaledTranslation(
            SHADOW_PX / 72, -SHADOW_PX / 72, fig.dpi_scale_trans
        )
        ax.bar(
            offset, value, width=bar_width,
            color=SHADOW_COLOR, edgecolor="none", linewidth=0,
            alpha=SHADOW_ALPHA, zorder=2, transform=shadow_transform,
        )
        ax.bar(
            offset, value, width=bar_width,
            color=color, edgecolor=edge, linewidth=lw, zorder=3,
        )
        if std > 0:
            ax.errorbar(
                offset, value, yerr=std,
                fmt="none", ecolor="#222222", elinewidth=1.0,
                capsize=2.8, capthick=1.0, zorder=4,
            )

    half = (n_methods - 1) * spacing / 2 + bar_width
    ax.set_xlim(-half - 0.04, half + 0.04)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.8)

    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)

avg_ax = axes[-1]
for spine in ["top", "right", "left", "bottom"]:
    avg_ax.spines[spine].set_visible(True)
    avg_ax.spines[spine].set_color(AVERAGE_BORDER)
    avg_ax.spines[spine].set_linewidth(1.0)

axes[0].set_ylabel("Success Rate (%)", fontsize=10)
axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
for ax in axes[1:-1]:
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)

plt.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.14, wspace=0.10)
fig.canvas.draw()

# ── Legend ────────────────────────────────────────────────────────────────────
left_bbox  = axes[0].get_position()
right_bbox = axes[-1].get_position()

swatch_w        = 0.018
swatch_gap      = 0.006
item_gap        = 0.034
char_w          = 0.0102
side_pad        = 0.016

item_widths = [swatch_w + swatch_gap + char_w * len(lbl) for lbl in METHOD_LABELS]
legend_width = 2 * side_pad + sum(item_widths) + item_gap * (len(METHOD_LABELS) - 1)

fig_center_x   = (left_bbox.x0 + right_bbox.x1) / 2
x0_legend      = fig_center_x - legend_width / 2
legend_bottom  = left_bbox.y1 + 0.05
legend_height  = 0.07
legend_top     = legend_bottom + legend_height
legend_mid     = (legend_top + legend_bottom) / 2

fig.patches.append(
    mpatches.FancyBboxPatch(
        (x0_legend, legend_bottom), legend_width, legend_height,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor=LEGEND_BORDER, linewidth=0.8,
        transform=fig.transFigure, clip_on=False, zorder=10,
    )
)

swatch_h = legend_height * 0.55
cursor = x0_legend + side_pad
for i, (color, edge, label) in enumerate(zip(COLORS, EDGE_COLORS, METHOD_LABELS)):
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (cursor, legend_mid - swatch_h / 2),
            swatch_w, swatch_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor=edge, linewidth=0.8,
            transform=fig.transFigure, clip_on=False, zorder=11,
        )
    )
    fig.text(
        cursor + swatch_w + swatch_gap, legend_mid, label,
        ha="left", va="center", fontsize=9,
        transform=fig.transFigure, zorder=12,
    )
    cursor += item_widths[i] + item_gap

# ── Panel labels ───────────────────────────────────────────────────────────────
fig.canvas.draw()
for ax, sublabel in zip(axes, all_labels):
    bbox = ax.get_position()
    x_center = bbox.x0 + bbox.width / 2
    weight = "bold" if sublabel == "Average" else "normal"
    fig.text(
        x_center, bbox.y0 - 0.04, sublabel,
        ha="center", va="center",
        fontsize=9.5, fontweight=weight,
        transform=fig.transFigure, zorder=6,
    )

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "static", "images")
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "results_bar.svg"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT_DIR, "results_bar.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("Saved results_bar.svg and results_bar.png to", OUT_DIR)
