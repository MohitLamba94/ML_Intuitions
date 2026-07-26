"""Generate figures for floating_precison.md.

Outputs (all .jpg, per repo convention):
  assets/float_formats_bitlayout.jpg     -- sign/exponent/mantissa bit strips for
                                            FP32, FP16, BF16, FP8-E4M3, FP8-E5M2, FP4
  assets/quant_scaling_granularity.jpg   -- per-tensor vs per-channel vs per-block
                                            (microscaling) scale sharing on a matrix
  assets/precision_memory_tradeoff.jpg   -- memory to hold a 70B model vs bit-width

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_precision_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Same soft, brand-neutral palette used across the repo's figures.
C = {
    "blue":  "#6C8EBF",
    "purple":"#9673A6",
    "green": "#82B366",
    "orange":"#D6795B",
    "grey":  "#555555",
    "lgrey": "#EDEDED",
    "yellow":"#E8B84B",
    "text":  "#222222",
}

DPI = 150


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# Figure 1: bit layouts -- sign / exponent / mantissa strips, width = #bits
# ---------------------------------------------------------------------------
def fig_bitlayout():
    # name, exponent bits, mantissa bits, right-hand annotation
    fmts = [
        ("FP32",       8, 23, "~1e-38 .. 3.4e38"),
        ("FP16",       5, 10, "6e-5 .. 65504"),
        ("BF16",       8,  7, "~1e-38 .. 3.4e38"),
        ("FP8 E4M3",   4,  3, "~2e-3 .. 448"),
        ("FP8 E5M2",   5,  2, "~6e-5 .. 57344"),
        ("FP4 E2M1",   2,  1, "0.5 .. 6"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    rh = 0.62              # row height
    gap = 0.42            # vertical gap between rows
    for i, (name, e, m, rng) in enumerate(fmts):
        y = (len(fmts) - 1 - i) * (rh + gap)
        total = 1 + e + m
        # sign (1 bit) | exponent (e) | mantissa (m); short labels fit any width
        segs = [(0, 1, C["orange"], "S"),
                (1, e, C["blue"],   f"E{e}"),
                (1 + e, m, C["green"], f"M{m}")]
        for x0, w, col, lab in segs:
            ax.add_patch(Rectangle((x0, y), w, rh, facecolor=col,
                                   edgecolor="white", linewidth=1.4))
            ax.text(x0 + w / 2, y + rh / 2, lab, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        # format name on the left
        ax.text(-0.4, y + rh / 2, name, ha="right", va="center",
                fontsize=10.5, color=C["text"], fontweight="bold")
        # total bits + range on the right
        ax.text(total + 0.5, y + rh / 2, f"{total} bits   ·   {rng}",
                ha="left", va="center", fontsize=9, color=C["grey"])

    ax.set_xlim(-6.5, 44)
    ax.set_ylim(-0.3, len(fmts) * (rh + gap))
    ax.axis("off")
    # legend of the three fields
    handles = [Rectangle((0, 0), 1, 1, facecolor=C["orange"]),
               Rectangle((0, 0), 1, 1, facecolor=C["blue"]),
               Rectangle((0, 0), 1, 1, facecolor=C["green"])]
    ax.legend(handles, ["sign  (±)", "exponent  (range)", "mantissa  (precision)"],
              loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=9.5)
    fig.suptitle("Every float = sign · exponent · mantissa "
                 "(bar width = number of bits stored)",
                 fontsize=12.5, color=C["text"], y=0.99)
    fig.tight_layout()
    save(fig, "float_formats_bitlayout.jpg")


# ---------------------------------------------------------------------------
# Figure 2: scale-sharing granularity (per-tensor / per-channel / per-block)
# ---------------------------------------------------------------------------
def fig_granularity():
    rng = np.random.default_rng(3)
    n = 8
    M = rng.uniform(0.05, 1.0, size=(n, n))
    M[2, 5] = 9.0          # a single large outlier
    M[6, 1] = 6.0          # a second outlier in a different column

    panels = [
        ("Per-tensor\n(1 scale for everything)", "tensor"),
        ("Per-channel\n(1 scale per column)", "channel"),
        ("Microscaling / per-block\n(1 scale per small block)", "block"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
    for ax, (title, kind) in zip(axes, panels):
        ax.imshow(np.log10(M), cmap="YlGnBu", vmin=-1.3, vmax=1.0)
        ax.set_title(title, fontsize=10.5, color=C["text"])
        ax.set_xticks([]); ax.set_yticks([])
        # mark the outliers
        for (r, c) in [(2, 5), (6, 1)]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=C["orange"], linewidth=2.4))

        def box(x0, y0, w, h, col):
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                                   edgecolor=col, linewidth=2.0))
        if kind == "tensor":
            box(0, 0, n, n, C["purple"])
        elif kind == "channel":
            for c in range(n):
                box(c, 0, 1, n, C["purple"])
        else:  # block: 2x2 blocks
            for r in range(0, n, 2):
                for c in range(0, n, 2):
                    box(c, r, 2, 2, C["purple"])
    axes[0].text(0.5, -0.12,
                 "One outlier forces the shared scale to be huge → the many small "
                 "values collapse to a few coarse levels (orange = outliers).",
                 transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=9.2, color=C["grey"])
    fig.suptitle("Finer scale-sharing isolates outliers so most values keep "
                 "their precision", fontsize=12.5, color=C["text"], y=1.02)
    fig.tight_layout()
    save(fig, "quant_scaling_granularity.jpg")


# ---------------------------------------------------------------------------
# Figure 3: memory to hold a 70B-parameter model vs bit-width
# ---------------------------------------------------------------------------
def fig_memory():
    labels = ["FP32\n(32-bit)", "FP16/BF16\n(16-bit)", "FP8/INT8\n(8-bit)",
              "INT4/NF4\n(4-bit)", "2-bit"]
    bits = np.array([32, 16, 8, 4, 2])
    params = 70e9
    gb = params * bits / 8 / 1e9   # bytes -> GB
    cols = [C["grey"], C["blue"], C["green"], C["orange"], C["purple"]]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, gb, color=cols, edgecolor="white", width=0.66)
    for xi, g in zip(x, gb):
        ax.text(xi, g + 4, f"{g:.0f} GB", ha="center", fontsize=10,
                color=C["text"], fontweight="bold")
    # reference line: one 80GB H100/A100
    ax.axhline(80, color=C["orange"], linestyle="--", linewidth=1.3)
    ax.text(len(labels) - 0.5, 86, "one 80 GB GPU", ha="right",
            fontsize=9, color=C["orange"])
    ax.set_ylim(0, 300)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, color=C["text"])
    ax.set_ylabel("memory for weights (GB)", fontsize=10.5, color=C["text"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=C["grey"])
    fig.suptitle("A 70B model: halving the bits halves the memory",
                 fontsize=12.5, color=C["text"], y=0.98)
    fig.tight_layout()
    save(fig, "precision_memory_tradeoff.jpg")


if __name__ == "__main__":
    fig_bitlayout()
    fig_granularity()
    fig_memory()
