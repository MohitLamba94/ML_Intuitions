"""Generate figures for training_llms/003_NormalisationLayers.md.

Outputs (all .jpg, per repo convention):
  assets/norm_axes_ln_vs_bn.jpg   -- which axis LayerNorm vs BatchNorm reduce over
  assets/norm_ln_vs_rmsnorm.jpg   -- LayerNorm vs RMSNorm op flow (dropped boxes greyed)
  assets/norm_flop_vs_runtime.jpg -- Table 1 of Ivanov et al. 2020: %FLOP vs %runtime

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_norm_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_pe_figs.py).
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
# 1. Which axis does each norm reduce over?
#    A grid of (batch samples) x (feature dimensions). LayerNorm normalises
#    across the features of ONE sample (a row); BatchNorm across the batch for
#    ONE feature (a column).
# ---------------------------------------------------------------------------
def fig_axes_ln_vs_bn():
    n_batch, n_feat = 5, 7
    fig, (axL, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    def draw_grid(ax, highlight, title, sub):
        for r in range(n_batch):
            for c in range(n_feat):
                if highlight == "row" and r == n_batch - 2:
                    fc = C["blue"]
                elif highlight == "col" and c == 2:
                    fc = C["orange"]
                else:
                    fc = C["lgrey"]
                cell = FancyBboxPatch((c, n_batch - 1 - r), 0.9, 0.9,
                                      boxstyle="round,pad=0.02,rounding_size=0.05",
                                      fc=fc, ec=C["grey"], lw=1.0)
                ax.add_patch(cell)
        ax.set_xlim(-1.4, n_feat + 0.2)
        ax.set_ylim(-1.1, n_batch + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        # axis labels
        ax.annotate("", xy=(n_feat + 0.05, -0.55), xytext=(0, -0.55),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.6))
        ax.text(n_feat / 2.0, -1.0, "feature dimension  $i$  (0 … $d-1$)",
                ha="center", fontsize=11, color=C["text"])
        ax.annotate("", xy=(-0.75, n_batch + 0.05), xytext=(-0.75, 0),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.6))
        ax.text(-1.15, n_batch / 2.0, "batch  (samples / tokens)",
                ha="center", va="center", rotation=90, fontsize=11, color=C["text"])
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=C["text"])
        ax.text(n_feat / 2.0, n_batch + 0.15, sub, ha="center", fontsize=10.5,
                style="italic", color=C["grey"])

    draw_grid(axL, "row",
              "LayerNorm / RMSNorm",
              "normalise across all features of one sample (a row) — batch-independent")
    draw_grid(axB, "col",
              "BatchNorm",
              "normalise one feature across the whole batch (a column) — couples samples")
    fig.tight_layout()
    save(fig, "norm_axes_ln_vs_bn.jpg")


# ---------------------------------------------------------------------------
# 2. LayerNorm vs RMSNorm: the op pipeline, with the two ops RMSNorm drops
#    shown greyed / crossed out.
# ---------------------------------------------------------------------------
def fig_ln_vs_rmsnorm():
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box_w, box_h = 1.9, 0.9

    def box(x, y, text, fc, ec, alpha=1.0, txt_col=None):
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     fc=fc, ec=ec, lw=1.4, alpha=alpha))
        ax.text(x + box_w / 2, y + box_h / 2, text, ha="center", va="center",
                fontsize=10.5, color=txt_col or C["text"])

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y + box_h / 2), (x1, y + box_h / 2),
                     arrowstyle="-|>", mutation_scale=16, color=C["grey"], lw=1.5))

    xs = [0.3, 2.6, 4.9, 7.2, 9.5]

    # --- LayerNorm row (top) ---
    yL = 3.5
    ax.text(0.3, yL + 1.15, "LayerNorm", fontsize=13, fontweight="bold",
            color=C["text"])
    box(xs[0], yL, "input  $x$", C["lgrey"], C["grey"])
    box(xs[1], yL, "subtract mean\n$x-\\mu$", C["yellow"], C["grey"])
    box(xs[2], yL, "divide by std\n$/\\sqrt{\\sigma^2+\\epsilon}$", C["blue"], C["grey"],
        txt_col="white")
    box(xs[3], yL, "scale\n$\\times\\,g$", C["green"], C["grey"], txt_col="white")
    box(xs[4], yL, "shift\n$+\\,b$", C["yellow"], C["grey"])
    for i in range(4):
        arrow(xs[i] + box_w, xs[i + 1], yL)

    # --- RMSNorm row (bottom) ---
    yR = 0.9
    ax.text(0.3, yR + 1.15, "RMSNorm", fontsize=13, fontweight="bold",
            color=C["text"])
    box(xs[0], yR, "input  $x$", C["lgrey"], C["grey"])
    # dropped: subtract mean
    box(xs[1], yR, "subtract mean", "white", "#cccccc", alpha=0.6,
        txt_col="#bbbbbb")
    box(xs[2], yR, "divide by RMS\n$/\\sqrt{\\overline{x^2}+\\epsilon}$", C["blue"],
        C["grey"], txt_col="white")
    box(xs[3], yR, "scale\n$\\times\\,g$", C["green"], C["grey"], txt_col="white")
    # dropped: shift/bias
    box(xs[4], yR, "shift  $+b$", "white", "#cccccc", alpha=0.6, txt_col="#bbbbbb")
    # cross out the two dropped boxes
    for xi in (xs[1], xs[4]):
        ax.plot([xi + 0.15, xi + box_w - 0.15], [yR + box_h - 0.12, yR + 0.12],
                color=C["orange"], lw=2.2)
    # solid arrows through the kept path, dashed where a box is skipped
    ax.add_patch(FancyArrowPatch((xs[0] + box_w, yR + box_h / 2),
                 (xs[2], yR + box_h / 2), arrowstyle="-|>", mutation_scale=16,
                 color=C["orange"], lw=1.6, linestyle=(0, (4, 3)),
                 connectionstyle="arc3,rad=-0.28"))
    arrow(xs[2] + box_w, xs[3], yR)
    ax.text((xs[0] + xs[2]) / 2 + 0.4, yR - 0.35, "skip mean",
            ha="center", fontsize=9, style="italic", color=C["orange"])

    fig.tight_layout()
    save(fig, "norm_ln_vs_rmsnorm.jpg")


# ---------------------------------------------------------------------------
# 3. Table 1 of Ivanov et al. 2020: normalisation is a tiny slice of FLOPs but
#    a big slice of runtime -> memory-bandwidth bound.
# ---------------------------------------------------------------------------
def fig_flop_vs_runtime():
    classes = ["Tensor contraction\n(matmuls)", "Statistical\nnormalization",
               "Element-wise"]
    flop = [99.80, 0.17, 0.03]
    runtime = [61.0, 25.5, 13.5]
    x = np.arange(len(classes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    b1 = ax.bar(x - w / 2, flop, w, label="% of FLOPs", color=C["blue"])
    b2 = ax.bar(x + w / 2, runtime, w, label="% of runtime", color=C["orange"])

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5,
                    f"{h:g}%", ha="center", va="bottom", fontsize=10,
                    color=C["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("share of a BERT training iteration", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("Norms are ~0.17% of the FLOPs but ~25% of the runtime\n"
                 "they are memory-bandwidth bound, not compute bound",
                 fontsize=12.5, fontweight="bold", color=C["text"])
    ax.legend(fontsize=11, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # highlight the normalization gap
    ax.annotate("tiny compute,\nlarge wall-clock cost",
                xy=(1 + w / 2, 25.5), xytext=(1.55, 55),
                fontsize=9.5, color=C["text"], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"],
                          lw=1.0, alpha=0.92),
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.4))
    fig.tight_layout()
    save(fig, "norm_flop_vs_runtime.jpg")


if __name__ == "__main__":
    fig_axes_ln_vs_bn()
    fig_ln_vs_rmsnorm()
    fig_flop_vs_runtime()
    print("done.")
