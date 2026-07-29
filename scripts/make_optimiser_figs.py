"""Generate figures for optimisers.md.

Outputs (all .jpg, per repo convention):
  assets/opt_ravine_sgd.jpg       -- ill-conditioned bowl + plain-GD zig-zag
  assets/opt_sgd_vs_momentum.jpg  -- same bowl: plain GD vs momentum trajectory
  assets/opt_adam_decomposition.jpg -- Adam step = direction (m) x per-coord scale (1/sqrt v)
  assets/opt_l2_vs_decoupled.jpg  -- coupled L2 vs decoupled weight decay schematic

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_optimiser_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# A soft, consistent, brand-neutral palette (matches make_norm_figs.py).
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "yellow": "#E8B84B",
    "text":   "#222222",
}

DPI = 150


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# The shared ill-conditioned quadratic: f(x,y) = 0.5*(a x^2 + b y^2).
# a >> b  ->  a long, narrow valley (steep in x, flat in y). This is the
# canonical picture that motivates BOTH momentum and adaptive methods.
# ---------------------------------------------------------------------------
A, B = 20.0, 1.0  # curvatures: steep along x, shallow along y


def f(x, y):
    return 0.5 * (A * x ** 2 + B * y ** 2)


def grad(x, y):
    return np.array([A * x, B * y])


def run_gd(lr, steps, start, momentum=0.0):
    """Plain / heavy-ball GD. Returns the (steps+1, 2) trajectory."""
    p = np.array(start, dtype=float)
    v = np.zeros(2)
    traj = [p.copy()]
    for _ in range(steps):
        g = grad(*p)
        v = momentum * v + g
        p = p - lr * v
        traj.append(p.copy())
    return np.array(traj)


def _contours(ax):
    xs = np.linspace(-1.15, 1.15, 400)
    ys = np.linspace(-2.6, 2.6, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    ax.contour(X, Y, Z, levels=np.linspace(0.3, 12, 12),
               colors=C["grey"], linewidths=0.6, alpha=0.5)
    ax.plot(0, 0, marker="*", ms=16, color=C["green"], zorder=5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_traj(ax, traj, color, label):
    ax.plot(traj[:, 0], traj[:, 1], "-o", color=color, ms=3.5, lw=1.6,
            label=label, zorder=4)
    ax.plot(traj[0, 0], traj[0, 1], marker="o", ms=9, color=color,
            markeredgecolor="white", zorder=6)


# ---------------------------------------------------------------------------
# 1. Plain GD zig-zags across a ravine.
# ---------------------------------------------------------------------------
def fig_ravine_sgd():
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    _contours(ax)
    # lr near the stability limit of the steep direction -> zig-zag
    traj = run_gd(lr=0.09, steps=40, start=(1.0, 2.3))
    _plot_traj(ax, traj, C["orange"], "plain GD")
    ax.set_title("Plain gradient descent on an\nill-conditioned valley",
                 fontsize=12.5, fontweight="bold", color=C["text"])
    ax.annotate("steep, narrow\ndirection", xy=(0.55, 0.05), xytext=(0.15, -1.7),
                fontsize=9.5, color=C["grey"], ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.2))
    ax.text(0.0, 2.42, "flat, long direction  (where we actually want to go)",
            ha="center", fontsize=9.5, style="italic", color=C["grey"])
    ax.text(0.0, -2.48, "big steps waste themselves bouncing across the valley",
            ha="center", fontsize=9.5, color=C["orange"])
    fig.tight_layout()
    save(fig, "opt_ravine_sgd.jpg")


# ---------------------------------------------------------------------------
# 2. Momentum damps the zig-zag and accelerates down the valley.
# ---------------------------------------------------------------------------
def fig_sgd_vs_momentum():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 6.2))
    for ax in (axL, axR):
        _contours(ax)

    traj_gd = run_gd(lr=0.09, steps=40, start=(1.0, 2.3))
    _plot_traj(axL, traj_gd, C["orange"], "plain GD")
    axL.set_title("Plain GD", fontsize=12.5, fontweight="bold", color=C["text"])
    axL.text(0.0, -2.48, "oscillates; crawls along the valley",
             ha="center", fontsize=9.5, color=C["orange"])

    traj_mom = run_gd(lr=0.045, steps=40, start=(1.0, 2.3), momentum=0.85)
    _plot_traj(axR, traj_mom, C["blue"], "momentum ($\\gamma=0.85$)")
    axR.set_title("SGD + Momentum ($\\gamma=0.85$)", fontsize=12.5,
                  fontweight="bold", color=C["text"])
    axR.text(0.0, -2.48, "cross-valley wobble cancels; along-valley speed builds",
             ha="center", fontsize=9.5, color=C["blue"])

    fig.tight_layout()
    save(fig, "opt_sgd_vs_momentum.jpg")


# ---------------------------------------------------------------------------
# 3. Adam's step decomposed: a momentum DIRECTION rescaled PER-COORDINATE by
#    1/sqrt(v). A noisy/steep coordinate gets shrunk; a quiet one gets boosted.
# ---------------------------------------------------------------------------
def fig_adam_decomposition():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box_w, box_h = 2.35, 1.5

    def box(x, y, title, sub, fc, txt_col=None):
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     fc=fc, ec=C["grey"], lw=1.4))
        ax.text(x + box_w / 2, y + box_h * 0.68, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=txt_col or C["text"])
        ax.text(x + box_w / 2, y + box_h * 0.28, sub, ha="center", va="center",
                fontsize=9.5, color=txt_col or C["text"])

    def arrow(x0, x1, y, txt):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=18, color=C["grey"], lw=1.6))
        ax.text((x0 + x1) / 2, y + 0.22, txt, ha="center", fontsize=11,
                color=C["grey"])

    y = 1.6
    xs = [0.2, 4.35, 8.5]
    box(xs[0], y, "first moment  $\\hat m_t$",
        "EMA of gradients\n= smoothed direction", C["blue"], "white")
    box(xs[1], y, "per-coord scale",
        "$1/(\\sqrt{\\hat v_t}+\\epsilon)$\nbig grads shrink", C["yellow"])
    box(xs[2], y, "Adam step",
        "$\\eta\\,\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon)$", C["green"], "white")
    arrow(xs[0] + box_w, xs[1], y + box_h / 2, "$\\times$")
    arrow(xs[1] + box_w, xs[2], y + box_h / 2, "$=$")

    ax.text(5.75, 4.25,
            "Adam = momentum for the DIRECTION  $\\times$  RMSprop for the STEP SIZE",
            ha="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.text(5.75, 0.7,
            "Each coordinate gets its own learning rate: noisy / steep coords are "
            "damped, quiet coords are amplified.",
            ha="center", fontsize=9.5, style="italic", color=C["grey"])
    fig.tight_layout()
    save(fig, "opt_adam_decomposition.jpg")


# ---------------------------------------------------------------------------
# 4. L2 (coupled) vs decoupled weight decay (AdamW).
#    Coupled: decay term goes THROUGH the 1/sqrt(v) rescaling -> uneven shrink.
#    Decoupled: decay is a clean, uniform pull toward 0, added after.
# ---------------------------------------------------------------------------
def fig_l2_vs_decoupled():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for ax in (axL, axR):
        ax.set_xlim(0, 6.4)
        ax.set_ylim(0, 4.7)
        ax.axis("off")

    bw, bh = 1.75, 0.85

    def box(ax, x, y, text, fc, txt_col=None):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     fc=fc, ec=C["grey"], lw=1.3))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                fontsize=10, color=txt_col or C["text"])

    def arr(ax, x0, y0, x1, y1, color=C["grey"], style="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=15, color=color, lw=1.5, linestyle=style))

    # ---- LEFT: Adam + L2 (coupled) ----
    axL.set_title("Adam + L2  (coupled — the buggy default)",
                  fontsize=12, fontweight="bold", color=C["orange"])
    box(axL, 0.4, 3.4, "loss grad $g_t$", C["lgrey"])
    box(axL, 2.6, 3.4, "$+\\ \\lambda\\theta$  (L2)", C["orange"], "white")
    box(axL, 1.5, 1.9, "adaptive rescale\n$/(\\sqrt{\\hat v_t}+\\epsilon)$", C["blue"], "white")
    box(axL, 1.5, 0.5, "update $\\theta$", C["green"], "white")
    arr(axL, 2.15, 3.82, 2.6, 3.82)
    arr(axL, 2.6 + bw / 2, 3.4, 1.5 + bw, 1.9 + bh)
    arr(axL, 1.5 + bw / 2, 1.9, 1.5 + bw / 2, 0.5 + bh)
    axL.text(3.2, 1.35,
             "decay divided by $\\sqrt{\\hat v_t}$ too:\nlarge-grad weights decay LESS",
             fontsize=9, color=C["orange"], ha="center", style="italic")

    # ---- RIGHT: AdamW (decoupled) ----
    axR.set_title("AdamW  (decoupled — correct)",
                  fontsize=12, fontweight="bold", color=C["green"])
    box(axR, 0.4, 3.4, "loss grad $g_t$", C["lgrey"])
    box(axR, 0.4, 1.9, "adaptive rescale\n$/(\\sqrt{\\hat v_t}+\\epsilon)$", C["blue"], "white")
    box(axR, 3.9, 1.9, "$-\\ \\eta\\lambda\\theta$\nuniform shrink", C["purple"], "white")
    box(axR, 2.1, 0.5, "update $\\theta$", C["green"], "white")
    arr(axR, 0.4 + bw / 2, 3.4, 0.4 + bw / 2, 1.9 + bh)
    arr(axR, 0.4 + bw, 1.9 + bh / 2, 2.1 + bw / 2, 0.5 + bh)
    arr(axR, 3.9 + bw / 2, 1.9, 2.1 + bw / 2, 0.5 + bh, color=C["purple"])
    axR.text(4.75, 1.25, "decay added\nseparately: same\npull on every weight",
             fontsize=9, color=C["purple"], ha="center", style="italic")

    fig.tight_layout()
    save(fig, "opt_l2_vs_decoupled.jpg")


if __name__ == "__main__":
    fig_ravine_sgd()
    fig_sgd_vs_momentum()
    fig_adam_decomposition()
    fig_l2_vs_decoupled()
    print("done.")
