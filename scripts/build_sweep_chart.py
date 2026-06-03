#!/usr/bin/env python3
"""Bar chart of the base-policy prompt/guidance sweep: model-success vs usable rates.

Used to pick which configuration generates the cleanest training data per task.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

mpl.rcParams["font.family"] = "monospace"

# config label -> (model success %, usable %); chosen flag
SWEEP = {
    "Cube": [
        ("g3\nbaseline", 40, 30, False),
        ("g3\nblock",    70, 40, False),
        ("g3\nwooden",   50, 30, False),
        ("g5\nbaseline", 80, 50, False),
        ("g5\nblock",    80, 50, True),
        ("g5\nwooden",   70, 50, False),
    ],
    "Tomato": [
        ("g3\nbaseline", 30, 30, False),
        ("g3\nred ball", 70, 40, True),
        ("g3\nrubber",   40, 30, False),
        ("g5\nbaseline", 50, 20, False),
        ("g5\nred ball", 20, 20, False),
        ("g5\nrubber",   30, 30, False),
    ],
}

SUCCESS_COLOR = "#AFC3C5"
USABLE_COLOR  = "#F9CF48"
EDGE          = "#000000"
GRID_COLOR    = "#d0d0d0"
SPINE_COLOR   = "#b0b0b0"
CHOSEN_TEXT   = "#1f9d55"
CHOSEN_BOX    = "#b7e4c7"

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
fig.patch.set_facecolor("white")

bar_w = 0.38

for ax, (task, rows) in zip(axes, SWEEP.items()):
    labels = [r[0] for r in rows]
    succ   = [r[1] for r in rows]
    usable = [r[2] for r in rows]
    chosen = [r[3] for r in rows]
    x = np.arange(len(rows))

    ax.bar(x - bar_w / 2, succ, width=bar_w, color=SUCCESS_COLOR,
           edgecolor=EDGE, linewidth=0.8, zorder=3, label="Model success")
    ax.bar(x + bar_w / 2, usable, width=bar_w, color=USABLE_COLOR,
           edgecolor=EDGE, linewidth=0.8, zorder=3, label="Usable")

    ax.set_title(task, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.8)

    for xi, ch in zip(x, chosen):
        if ch:
            ax.add_patch(
                Rectangle(
                    (xi - 0.48, 0),
                    0.96,
                    100,
                    fill=False,
                    edgecolor=CHOSEN_BOX,
                    linewidth=2.2,
                    zorder=4,
                )
            )
            ax.text(xi, 94, "used", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=CHOSEN_TEXT, zorder=5)

axes[0].set_ylabel("Rate (%)", fontsize=10)
axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
axes[0].legend(loc="upper left", fontsize=8.5, frameon=True, edgecolor="#888888")

plt.tight_layout()

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "static", "images")
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "sweep_rates.svg"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT_DIR, "sweep_rates.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("Saved sweep_rates.svg/.png to", OUT_DIR)
