"""Figures for training_llms/004_Activations.md (gated activations: GELU, SwiGLU/GeGLU).

Run with the `general` conda env python. Saves .jpg into ../assets/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import erf

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

BLUE = "#2c6fbb"
ORANGE = "#e07b39"
GREEN = "#3a9d6b"
GREY = "#8a8a8a"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gelu(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def silu(x):  # Swish-1
    return x * sigmoid(x)


def relu(x):
    return np.maximum(0.0, x)


# ---------------------------------------------------------------------------
# Figure 1: activation shapes (ReLU vs GELU vs SiLU) + the soft gate view
# ---------------------------------------------------------------------------
def fig_activations():
    x = np.linspace(-5, 5, 600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: the three activation curves
    ax1.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax1.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax1.plot(x, relu(x), color=GREY, lw=2.2, label="ReLU  max(0, x)")
    ax1.plot(x, gelu(x), color=BLUE, lw=2.4, label=r"GELU  $x\,\Phi(x)$")
    ax1.plot(x, silu(x), color=ORANGE, lw=2.2, ls="--", label=r"SiLU/Swish  $x\,\sigma(x)$")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-1.0, 5)
    ax1.set_title("Smooth activations vs the ReLU kink", fontsize=12, fontweight="bold")
    ax1.set_xlabel("input  x")
    ax1.set_ylabel("output")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.annotate("small negative dip\n(non-monotonic)", xy=(-1.5, gelu(-1.5)),
                 xytext=(-4.6, 1.7), fontsize=8.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))

    # right: GELU as a soft gate -- the gate Phi(x) vs the hard ReLU gate
    hard_gate = (x > 0).astype(float)
    soft_gate = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))  # Phi(x)
    ax2.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax2.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax2.plot(x, hard_gate, color=GREY, lw=2.2, label=r"ReLU gate  $\mathbf{1}[x>0]$  (hard 0/1)")
    ax2.plot(x, soft_gate, color=BLUE, lw=2.4, label=r"GELU gate  $\Phi(x)$  (soft 0$\to$1)")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-0.1, 1.15)
    ax2.set_title("GELU keeps a fraction $\\Phi(x)$ of the input", fontsize=12, fontweight="bold")
    ax2.set_xlabel("input  x")
    ax2.set_ylabel("gate value (fraction of x kept)")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax2.annotate("ReLU decides abruptly", xy=(0.05, 0.5), xytext=(1.2, 0.35),
                 fontsize=8.5, color=GREY,
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))

    fig.suptitle(r"output $=x\cdot\mathrm{gate}(x)$:  ReLU uses a hard step, GELU/SiLU use a smooth probability",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "act_relu_gelu_silu.jpg")
    fig.savefig(out, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: standard FFN vs gated (GLU) FFN block diagram
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, tc="#111111"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, lw=1.5, color=color, zorder=2))


def fig_ffn():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ---- Standard FFN (left) ----
    ax1.set_title("Standard FFN  (2 matrices)", fontsize=12.5, fontweight="bold")
    _box(ax1, 3.5, 0.6, 3, 1.1, "input  x", "#eef3fa")
    _box(ax1, 3.5, 3.0, 3, 1.1, r"up-proj  $xW$", "#dbe7f6")
    _box(ax1, 3.5, 5.2, 3, 1.1, "activation\n(ReLU / GELU)", "#cfe6d8")
    _box(ax1, 3.5, 7.6, 3, 1.1, r"down-proj  $\cdot\,W_2$", "#dbe7f6")
    _arrow(ax1, 5, 1.7, 5, 3.0)
    _arrow(ax1, 5, 4.1, 5, 5.2)
    _arrow(ax1, 5, 6.3, 5, 7.6)
    ax1.text(5, 9.2, r"$\max(0,\,xW)\,W_2$", ha="center", fontsize=11, color="#333")
    ax1.text(6.9, 3.55, r"$d\!\to\!d_{ff}$", fontsize=8.5, color=GREY)
    ax1.text(6.9, 8.15, r"$d_{ff}\!\to\!d$", fontsize=8.5, color=GREY)

    # ---- Gated FFN (right) ----
    ax2.set_title("Gated FFN / GLU  (3 matrices)", fontsize=12.5, fontweight="bold")
    _box(ax2, 3.5, 0.6, 3, 1.1, "input  x", "#eef3fa")
    # two parallel up-projections
    _box(ax2, 0.8, 3.0, 3.2, 1.1, r"gate proj  $xW$", "#dbe7f6")
    _box(ax2, 6.0, 3.0, 3.2, 1.1, r"value proj  $xV$", "#f6e4d6")
    _box(ax2, 0.8, 5.2, 3.2, 1.1, "activation\n(GELU / Swish)", "#cfe6d8")
    # split
    _arrow(ax2, 4.5, 1.7, 2.4, 3.0)
    _arrow(ax2, 5.5, 1.7, 7.6, 3.0)
    _arrow(ax2, 2.4, 4.1, 2.4, 5.2)
    # multiply node
    circ = plt.Circle((5.0, 6.6), 0.42, facecolor="#ffe08a", edgecolor="#8a6d1f",
                      lw=1.5, zorder=4)
    ax2.add_patch(circ)
    ax2.text(5.0, 6.6, r"$\odot$", ha="center", va="center", fontsize=15, zorder=5)
    _arrow(ax2, 2.4, 6.3, 4.6, 6.6)
    _arrow(ax2, 7.6, 4.1, 7.6, 6.6)
    _arrow(ax2, 7.6, 6.6, 5.42, 6.6)
    _box(ax2, 3.5, 8.0, 3, 1.1, r"down-proj  $\cdot\,W_2$", "#dbe7f6")
    _arrow(ax2, 5.0, 7.02, 5.0, 8.0)
    ax2.text(5.0, 9.55, r"$(\mathrm{act}(xW)\;\odot\;xV)\,W_2$", ha="center",
             fontsize=10.5, color="#333")
    ax2.text(2.4, 2.55, "gated branch", ha="center", fontsize=8.5, color=BLUE)
    ax2.text(7.6, 2.55, "linear branch", ha="center", fontsize=8.5, color=ORANGE)

    fig.suptitle("The gate: one branch is squashed by an activation, then multiplied element-wise into a plain linear copy",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "act_ffn_vs_gated_ffn.jpg")
    fig.savefig(out, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 3: Shazeer Table 1 log-perplexity bar chart
# ---------------------------------------------------------------------------
def fig_results():
    names = ["ReLU", "GELU", "Swish", "GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU"]
    ppl = [1.997, 1.983, 1.994, 1.982, 1.960, 1.953, 1.942, 1.944]
    gated = [False, False, False, True, True, True, True, True]
    colors = [ORANGE if g else GREY for g in gated]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(names, ppl, color=colors, edgecolor="#333333", lw=0.8, width=0.68)
    ax.set_ylim(1.90, 2.01)
    ax.set_ylabel("heldout log-perplexity  (lower is better)")
    ax.set_title("GLU variants beat non-gated baselines  (Shazeer 2020, Table 1, 65k steps)",
                 fontsize=12, fontweight="bold")
    baseline = ppl[0]
    ax.axhline(baseline, color=GREY, ls="--", lw=1.0)
    ax.text(1.5, baseline + 0.0016, "ReLU baseline", fontsize=8.5, color=GREY, ha="center")
    for b, v in zip(bars, ppl):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.3f}",
                ha="center", fontsize=8.5)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREY, label="non-gated (2 matrices)"),
                       Patch(facecolor=ORANGE, label="gated GLU (3 matrices)")],
              loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(ASSETS, "act_glu_perplexity.jpg")
    fig.savefig(out, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_activations()
    fig_ffn()
    fig_results()
