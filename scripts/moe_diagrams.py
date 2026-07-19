"""Generate the conceptual diagrams used by mixture-of-experts.md.

Run with the repo's conda env python:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/moe_diagrams.py
All figures are written as .jpg into ../assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

BG = "#f7f8fa"
INK = "#20242b"
MUTED = "#69707d"
BLUE = "#2878b5"
CYAN = "#55a9bd"
GREEN = "#4c956c"
AMBER = "#d99b2b"
RED = "#c9574f"
PURPLE = "#7b61a8"
GREY = "#c3c8d0"


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", ASSETS / name)


def rbox(ax, x, y, w, h, text, fc, ec=None, fontsize=11, tc="white", lw=1.4, weight="bold"):
    ec = ec or fc
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=1.0,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fontsize, fontweight=weight)
    return box


def arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>", alpha=1.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, alpha=alpha, linestyle=ls,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# 1. Dense FFN vs sparse MoE layer
# ---------------------------------------------------------------------------
def dense_vs_moe():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=BG)

    # -- Dense --
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Dense FFN block", color=INK, fontsize=15, fontweight="bold", pad=14)
    ax.text(0.5, 1.0, "every token uses ALL parameters", ha="center",
            color=MUTED, fontsize=10)
    rbox(ax, 0.35, 0.06, 0.30, 0.10, "token x", CYAN, fontsize=12)
    rbox(ax, 0.30, 0.42, 0.40, 0.16, "FFN\n(one big MLP)", BLUE, fontsize=12)
    rbox(ax, 0.35, 0.84, 0.30, 0.10, "output y", GREEN, fontsize=12)
    arrow(ax, (0.5, 0.16), (0.5, 0.42))
    arrow(ax, (0.5, 0.58), (0.5, 0.84))
    ax.text(0.5, 0.32, "active = total\nparams", ha="center", color=RED,
            fontsize=9.5, style="italic")

    # -- MoE --
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Sparse MoE layer  (top-2 of 4)", color=INK, fontsize=15,
                 fontweight="bold", pad=14)
    ax.text(0.5, 1.0, "each token uses only k of N experts", ha="center",
            color=MUTED, fontsize=10)
    rbox(ax, 0.35, 0.05, 0.30, 0.09, "token x", CYAN, fontsize=12)
    rbox(ax, 0.34, 0.24, 0.32, 0.10, "Router  G(x)", PURPLE, fontsize=11)
    arrow(ax, (0.5, 0.14), (0.5, 0.24))

    ex_x = [0.06, 0.30, 0.54, 0.78]
    chosen = {1, 2}  # experts 2 and 3 selected
    gates = ["", "g=0.7", "g=0.3", ""]
    for i, x in enumerate(ex_x):
        on = i in chosen
        fc = AMBER if on else GREY
        tc = "white" if on else MUTED
        rbox(ax, x, 0.50, 0.16, 0.13, f"E{i+1}", fc, fontsize=12, tc=tc)
        # router -> expert
        arrow(ax, (0.5, 0.34), (x + 0.08, 0.50),
              color=INK if on else GREY, lw=1.7 if on else 1.0,
              alpha=1.0 if on else 0.5, ls="-" if on else (0, (3, 3)))
        if on:
            ax.text(x + 0.08, 0.655, gates[i], ha="center", color=RED,
                    fontsize=9.5, fontweight="bold")
            arrow(ax, (x + 0.08, 0.63), (0.5, 0.80), color=AMBER, lw=1.7)

    rbox(ax, 0.32, 0.80, 0.36, 0.11, "weighted sum  y", GREEN, fontsize=11.5)
    ax.text(0.5, 0.955, r"y = 0.7·E2(x) + 0.3·E3(x)", ha="center",
            color=INK, fontsize=10)
    ax.text(0.02, 0.44, "grey = not run\n(0 FLOPs)", ha="left", color=MUTED,
            fontsize=8.5, style="italic")

    save(fig, "moe_dense_vs_layer.jpg")


# ---------------------------------------------------------------------------
# 2. Active vs total parameters (Mixtral-style)
# ---------------------------------------------------------------------------
def active_vs_total():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=BG)

    # Left: stacked composition of total vs active
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_title("Where the parameters live  (Mixtral 8x7B)", color=INK,
                 fontsize=14, fontweight="bold", pad=12)

    shared = 1.3       # attention + embeddings + norms (approx, illustrative)
    per_expert = 5.7   # approx params per expert (illustrative)
    n_experts = 8
    k = 2

    # total bar
    ax.bar(0, shared, color=BLUE, width=0.6, label="shared (attn+emb)")
    bottom = shared
    for i in range(n_experts):
        c = AMBER if i < k else GREY
        ax.bar(0, per_expert, bottom=bottom, color=c, width=0.6,
               edgecolor="white", linewidth=1.0)
        bottom += per_expert
    # active bar
    ax.bar(1, shared, color=BLUE, width=0.6)
    ax.bar(1, per_expert * k, bottom=shared, color=AMBER, width=0.6,
           edgecolor="white", linewidth=1.0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TOTAL\n~47B params", "ACTIVE / token\n~13B params"],
                       fontsize=11)
    ax.set_ylabel("parameters (billions, illustrative)", color=INK, fontsize=10)
    ax.text(0.42, shared + per_expert * n_experts - 4, "all 8 experts\nheld in VRAM",
            ha="left", color=MUTED, fontsize=9)
    ax.text(1, shared + per_expert * k + 1.4, "only 2 experts\nrun (compute)",
            ha="center", color=MUTED, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(loc="center right", fontsize=8.5, frameon=False)

    # Right: the crisp takeaway
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Two numbers, two costs", color=INK, fontsize=14,
                 fontweight="bold", pad=12)
    rbox(ax, 0.08, 0.62, 0.84, 0.20,
         "TOTAL params  →  VRAM / memory\n(must store every expert)",
         BLUE, fontsize=12, weight="bold")
    rbox(ax, 0.08, 0.30, 0.84, 0.20,
         "ACTIVE params  →  compute / latency\n(FLOPs/token = a dense k-expert model)",
         AMBER, fontsize=11, weight="bold")
    ax.text(0.5, 0.14,
            "MoE buys a huge model's quality\nat a small model's compute — "
            "if you can afford the memory.",
            ha="center", color=MUTED, fontsize=10.5, style="italic")

    save(fig, "moe_active_vs_total.jpg")


# ---------------------------------------------------------------------------
# 3. Routing collapse vs balanced routing
# ---------------------------------------------------------------------------
def routing_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    experts = np.arange(1, 9)

    collapsed = np.array([0.46, 0.28, 0.14, 0.07, 0.03, 0.015, 0.005, 0.0])
    balanced = np.array([0.14, 0.13, 0.12, 0.125, 0.13, 0.12, 0.115, 0.12])

    for ax, data, title, col, sub in [
        (axes[0], collapsed, "Routing collapse (unstable)", RED,
         "a few experts win everything; the rest starve"),
        (axes[1], balanced, "Balanced routing (with aux loss)", GREEN,
         "load spread evenly → all capacity is used"),
    ]:
        ax.set_facecolor(BG)
        ax.bar(experts, data * 100, color=col, width=0.7,
               edgecolor="white", linewidth=1.0)
        ax.axhline(12.5, color=MUTED, ls="--", lw=1.2)
        ax.text(8.4, 13.5, "uniform 12.5%", color=MUTED, fontsize=8.5, ha="right")
        ax.set_title(title, color=INK, fontsize=14, fontweight="bold", pad=10)
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=9.5)
        ax.set_xlabel("expert index", color=INK, fontsize=10)
        ax.set_ylabel("% of tokens routed", color=INK, fontsize=10)
        ax.set_ylim(0, 50)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)

    save(fig, "moe_routing_collapse.jpg")


# ---------------------------------------------------------------------------
# 4. Expert parallelism + all-to-all
# ---------------------------------------------------------------------------
def expert_parallelism():
    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Expert parallelism: experts sharded across devices, tokens routed by all-to-all",
                 color=INK, fontsize=14, fontweight="bold", pad=14)

    devices = ["Device 0", "Device 1", "Device 2", "Device 3"]
    dx = [0.03, 0.28, 0.53, 0.78]
    dw = 0.19
    tok_cols = [CYAN, PURPLE, AMBER, GREEN]

    # top row: each device holds its own tokens + one expert
    for i, x in enumerate(dx):
        ax.add_patch(Rectangle((x, 0.08), dw, 0.84, facecolor="white",
                     edgecolor=GREY, linewidth=1.3, zorder=0))
        ax.text(x + dw / 2, 0.955, devices[i], ha="center", color=INK,
                fontsize=10.5, fontweight="bold")
        # local tokens (bottom)
        rbox(ax, x + 0.02, 0.12, dw - 0.04, 0.10, f"tokens (batch {i})",
             tok_cols[i], fontsize=8.5)
        # its expert (top)
        rbox(ax, x + 0.03, 0.72, dw - 0.06, 0.11, f"Expert {i}", BLUE, fontsize=10)

    # middle: the all-to-all shuffle band
    ax.add_patch(Rectangle((0.01, 0.36), 0.97, 0.26, facecolor="#eef1f5",
                 edgecolor=GREY, linewidth=1.0, zorder=0))
    ax.text(0.5, 0.585, "all-to-all DISPATCH  (each token → device holding its chosen expert)",
            ha="center", color=RED, fontsize=10, fontweight="bold")
    ax.text(0.5, 0.40, "all-to-all COMBINE  (results shuffled back to the token's home device)",
            ha="center", color=GREEN, fontsize=10, fontweight="bold")

    # crossing arrows to suggest the shuffle
    rng = np.random.default_rng(3)
    for i in range(4):
        for _ in range(2):
            j = rng.integers(0, 4)
            arrow(ax, (dx[i] + dw / 2, 0.225),
                  (dx[j] + dw / 2, 0.715),
                  color=tok_cols[i], lw=1.3, alpha=0.55)

    ax.text(0.5, 0.02,
            "Adding experts = adding devices; per-token compute stays fixed. "
            "The two all-to-alls are the main communication cost.",
            ha="center", color=MUTED, fontsize=9.5, style="italic")

    save(fig, "moe_expert_parallelism.jpg")


# ---------------------------------------------------------------------------
# 5. Load-balancing aux loss geometry: f_i * P_i minimized at uniform
# ---------------------------------------------------------------------------
def load_balancing_loss():
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    ax.set_facecolor(BG)

    # Two-expert intuition: fraction p to expert 1, (1-p) to expert 2.
    # If routing prob tracks the count, loss ~ N * (p^2 + (1-p)^2).
    p = np.linspace(0, 1, 400)
    loss = 2 * (p ** 2 + (1 - p) ** 2)  # N=2

    ax.plot(p, loss, color=BLUE, lw=2.6)
    ax.axvline(0.5, color=GREEN, ls="--", lw=1.5)
    ax.plot([0.5], [2 * (0.25 + 0.25)], "o", color=GREEN, ms=9, zorder=5)
    ax.annotate("minimum at\nbalanced (p=0.5)", (0.5, 1.0), (0.58, 1.35),
                color=GREEN, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("collapse\n(all to one expert)", (0.98, 1.98), (0.62, 1.75),
                color=RED, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))

    ax.set_title(r"Aux load-balancing loss  $\alpha N \sum_i f_i P_i$  (2 experts)",
                 color=INK, fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xlabel("fraction of tokens sent to expert 1", color=INK, fontsize=11)
    ax.set_ylabel("auxiliary loss (relative)", color=INK, fontsize=11)
    ax.set_ylim(0.8, 2.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.text(0.5, 0.86, "The loss is lowest when every expert gets an equal share, "
            "highest when one expert dominates.",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)

    save(fig, "moe_load_balancing.jpg")


if __name__ == "__main__":
    dense_vs_moe()
    active_vs_total()
    routing_collapse()
    expert_parallelism()
    load_balancing_loss()
    print("done")
