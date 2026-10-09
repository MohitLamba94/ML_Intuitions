"""Original diagrams for FlashAttention4.md; numbers from Zadouri et al., Tables 1 and 3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ASSETS = Path(__file__).resolve().parent.parent / "assets"
INK = "#202b36"
MUTED = "#516271"
GRID = "#d9dfe1"
MMA = "#2679ad"
SMEM = "#cb753c"
EXP = "#3c9a78"
LIGHT = "#f7f9f8"


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=175, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", ASSETS / name)


def resource_limits():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True, layout="constrained")
    panels = [
        ("Forward: M=N=d=128", [1024, 768, 1024], "MMA and exp tie"),
        ("Backward: M=N=d=128", [2560, 3328, 1024], "SMEM dominates"),
    ]
    for ax, (title, cycles, bottleneck) in zip(axes, panels):
        bars = ax.barh([2, 1, 0], cycles, color=[MMA, SMEM, EXP], height=0.55)
        ax.set_yticks([2, 1, 0], ["Tensor cores", "Shared memory", "Exponentials"])
        ax.set_xlim(0, 3950)
        ax.set_xticks([0, 1000, 2000, 3000], ["0", "1k", "2k", "3k"])
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, length=0)
        ax.tick_params(axis="y", pad=8)
        ax.set_title(title, loc="left", weight="bold", color=INK, fontsize=13, pad=15)
        for bar, value in zip(bars, cycles):
            ax.text(value + 55, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", color=INK, fontsize=10)
        ax.text(0.02, -0.18, bottleneck, transform=ax.transAxes, color=MUTED, fontsize=10)
        ax.set_xlabel("Estimated resource cycles (not total latency)", color=MUTED, labelpad=9)
    save(fig, "flashattention4_resource_limits.jpg")


def box(ax, x, y, width, height, label, color, text_color="white"):
    ax.add_patch(Rectangle((x, y), width, height, linewidth=0, facecolor=color))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=text_color, fontsize=10, weight="medium")


def arrow(ax, start, end, color=MUTED, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color, lw=1.7))


def cta_pair():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
    left, right = axes
    left.set_title("1-CTA mode: duplicated reads", loc="left", color=INK, weight="bold", fontsize=13)
    right.set_title("2-CTA mode: split operand and output", loc="left", color=INK, weight="bold", fontsize=13)

    for y, name in [(5.05, "CTA 0"), (1.9, "CTA 1")]:
        left.text(0.25, y + 1.2, name, color=INK, fontsize=10, weight="bold")
        box(left, 0.3, y, 2.45, 1.0, "full B in SMEM", SMEM)
        arrow(left, (2.85, y + 0.5), (3.65, y + 0.5))
        box(left, 3.75, y, 2.1, 1.0, "MMA", MMA)
        arrow(left, (5.95, y + 0.5), (6.75, y + 0.5))
        box(left, 6.85, y, 2.5, 1.0, "own rows in TMEM", EXP)
    left.text(0.3, 0.7, "Both CTAs fetch the same B tile", color=MUTED, fontsize=10)

    for y, name, half in [(5.05, "CTA 0", "first half of B"), (1.9, "CTA 1", "second half of B")]:
        right.text(0.25, y + 1.2, name, color=INK, fontsize=10, weight="bold")
        box(right, 0.3, y, 2.45, 1.0, half, SMEM)
        arrow(right, (2.85, y + 0.5), (3.65, y + 0.5))
        box(right, 3.75, y, 2.1, 1.0, "paired MMA", MMA)
        arrow(right, (5.95, y + 0.5), (6.75, y + 0.5))
        box(right, 6.85, y, 2.5, 1.0, "own rows in TMEM", EXP)
    arrow(right, (4.8, 3.0), (4.8, 4.95), color=SMEM, style="<->")
    right.text(5.05, 3.95, "dS exchange for dQ", va="center", color=SMEM, fontsize=9)
    right.text(0.3, 0.7, "Cluster DSMEM supplies the missing reduction half", color=MUTED, fontsize=10)
    fig.text(0.5, 0.005, "Conceptual operand sharing; arrows are not a literal instruction trace", ha="center", color=MUTED, fontsize=9)
    save(fig, "flashattention4_cta_pair.jpg")


if __name__ == "__main__":
    resource_limits()
    cta_pair()