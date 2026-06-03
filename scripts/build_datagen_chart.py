#!/usr/bin/env python3
"""Render the dataset-generation 'generated vs kept' bar chart for the writeup."""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mpl.rcParams["font.family"] = "monospace"

# task -> (generated, kept)
DATA = {
    "Cube":   (60, 30),
    "Tomato": (100, 24),
}

GEN_COLOR   = "#E4E4E4"
GEN_EDGE    = "#9a9a9a"
KEPT_COLOR  = "#F9CF48"
KEPT_EDGE   = "#000000"
GRID_COLOR  = "#d0d0d0"
SPINE_COLOR = "#b0b0b0"

labels = list(DATA.keys())
gen_vals  = [DATA[t][0] for t in labels]
kept_vals = [DATA[t][1] for t in labels]

x = range(len(labels))
bar_width = 0.45

fig, ax = plt.subplots(figsize=(5.2, 3.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for xi, gen, kept in zip(x, gen_vals, kept_vals):
    ax.bar(xi, gen, width=bar_width, color=GEN_COLOR, edgecolor=GEN_EDGE,
           linewidth=1.0, zorder=2)
    ax.bar(xi, kept, width=bar_width, color=KEPT_COLOR, edgecolor=KEPT_EDGE,
           linewidth=1.0, zorder=3)
    ax.text(xi, gen + 2, f"{gen} generated", ha="center", va="bottom",
            fontsize=8.5, color="#444444", zorder=4)
    ax.text(xi, kept / 2, f"{kept}\nkept", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#000000", zorder=5)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Trajectories", fontsize=10)
ymax = max(gen_vals) * 1.18
ax.set_ylim(0, ymax)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=8)

# Secondary axis: generation time. Each rollout costs ~136 s of GPU time.
SEC_PER_VIDEO = 136.0
HR_PER_VIDEO = SEC_PER_VIDEO / 3600.0
ax2 = ax.twinx()
ax2.set_ylim(0, ymax * HR_PER_VIDEO)
ax2.set_ylabel("Generation time (GPU-hr)", fontsize=10)
ax2.tick_params(axis="y", labelsize=8)

for spine in ["top"]:
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(SPINE_COLOR)
    ax.spines[spine].set_linewidth(0.8)
ax2.spines["right"].set_color(SPINE_COLOR)
ax2.spines["right"].set_linewidth(0.8)

legend_handles = [
    mpatches.Patch(facecolor=GEN_COLOR, edgecolor=GEN_EDGE, label="Generated"),
    mpatches.Patch(facecolor=KEPT_COLOR, edgecolor=KEPT_EDGE, label="Kept for BC"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
          frameon=True, edgecolor="#888888")

plt.tight_layout()

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "static", "images")
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "datagen_kept.svg"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT_DIR, "datagen_kept.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("Saved datagen_kept.svg/.png to", OUT_DIR)
