"""Generate figures for JEPA.md (Joint-Embedding Predictive Architectures, LeCun et al.).

Outputs (all .jpg, per repo convention):
  assets/jepa_arch_comparison.jpg  -- generative vs joint-embedding/contrastive vs JEPA, side by side
  assets/jepa_ijepa_masking.jpg    -- I-JEPA: one context block predicts several masked target blocks in rep space
  assets/jepa_collapse_energy.jpg  -- representation collapse (all points to one dot) vs a healthy spread
  assets/jepa_vicreg_terms.jpg     -- VICReg's three terms: variance, invariance, covariance
  assets/jepa_worldmodel_loop.jpg  -- V-JEPA 2-AC action-conditioned planning loop (MPC over latent predictions)

All schematic / illustrative (synthetic data), matplotlib only.

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_jepa_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Same brand-neutral palette as make_gated_attention_figs.py etc.
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "red":    "#C0504D",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "mgrey":  "#CCCCCC",
    "yellow": "#E8B84B",
    "text":   "#222222",
}
DPI = 150
np.random.seed(11)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def box(ax, x, y, w, h, label, fc, txt=None, fs=10, ec=None, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec=ec or C["grey"], lw=lw))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=txt or C["text"], zorder=5)


def arrow(ax, x1, y1, x2, y2, col=None, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=14,
                 color=col or C["grey"], lw=lw, zorder=1))


# ---------------------------------------------------------------------------
# 1. Three ways to do self-supervised learning, side by side.
#    Generative: encode -> decode -> reconstruct PIXELS.
#    Joint-embedding / contrastive: two encoders, pull positives / push negatives.
#    JEPA: context encoder + predictor -> match a TARGET REPRESENTATION (EMA, stop-grad).
# ---------------------------------------------------------------------------
def fig_arch_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    titles = ["Generative  (predict pixels)",
              "Joint-Embedding / Contrastive",
              "JEPA  (predict representations)"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(t, fontsize=12, color=C["text"], pad=6)

    # --- Panel A: generative ---
    ax = axes[0]
    box(ax, 1.0, 7.7, 8.0, 1.1, "input  x", C["lgrey"])
    box(ax, 2.5, 5.8, 5.0, 1.1, "encoder", C["blue"], txt="white")
    box(ax, 3.2, 3.9, 3.6, 1.0, "z", C["yellow"], fs=12)
    box(ax, 2.5, 2.0, 5.0, 1.1, "decoder", C["purple"], txt="white")
    box(ax, 1.0, 0.2, 8.0, 1.1, "reconstructed pixels  x̂", C["lgrey"])
    for (y1, y2) in [(7.7, 6.9), (5.8, 4.9), (3.9, 3.1), (2.0, 1.3)]:
        arrow(ax, 5.0, y1, 5.0, y2)
    ax.text(5.0, -0.6, "loss in PIXEL space\n(wastes capacity on unpredictable detail)",
            ha="center", va="top", fontsize=8.5, color=C["red"])

    # --- Panel B: contrastive ---
    ax = axes[1]
    box(ax, 0.6, 8.0, 3.6, 1.0, "view x", C["lgrey"])
    box(ax, 5.8, 8.0, 3.6, 1.0, "view x′", C["lgrey"])
    box(ax, 0.9, 5.9, 3.0, 1.1, "encoder", C["blue"], txt="white")
    box(ax, 6.1, 5.9, 3.0, 1.1, "encoder", C["blue"], txt="white")
    box(ax, 1.1, 4.0, 2.6, 1.0, "s", C["green"], fs=12)
    box(ax, 6.3, 4.0, 2.6, 1.0, "s′", C["green"], fs=12)
    arrow(ax, 2.4, 8.0, 2.4, 7.1); arrow(ax, 7.6, 8.0, 7.6, 7.1)
    arrow(ax, 2.4, 5.9, 2.4, 5.1); arrow(ax, 7.6, 5.9, 7.6, 5.1)
    ax.add_patch(FancyArrowPatch((3.7, 4.5), (6.3, 4.5), arrowstyle="<|-|>",
                 mutation_scale=13, color=C["green"], lw=2.0))
    ax.text(5.0, 3.4, "pull positives together", ha="center", fontsize=9, color=C["green"])
    ax.text(5.0, 1.9, "push NEGATIVES apart\n(needs many negative pairs)",
            ha="center", va="top", fontsize=9, color=C["red"])

    # --- Panel C: JEPA ---
    ax = axes[2]
    box(ax, 0.6, 8.0, 3.6, 1.0, "context x", C["lgrey"])
    box(ax, 5.8, 8.0, 3.6, 1.0, "target y", C["lgrey"])
    box(ax, 0.7, 5.9, 3.4, 1.1, "encoder\n$E_\\theta$", C["blue"], txt="white", fs=9)
    box(ax, 5.9, 5.9, 3.4, 1.1, "target enc.\n$E_{\\bar\\theta}$ (EMA)", C["orange"], txt="white", fs=8.5)
    box(ax, 1.2, 4.0, 2.4, 1.0, "$s_x$", C["green"], fs=12)
    box(ax, 6.4, 4.0, 2.4, 1.0, "$s_y$", C["green"], fs=12)
    box(ax, 1.0, 2.0, 2.8, 1.1, "predictor\n$P_\\phi$", C["purple"], txt="white", fs=9)
    box(ax, 1.05, 0.35, 2.7, 0.95, "$\\hat s_y$", C["yellow"], fs=12)
    arrow(ax, 2.4, 8.0, 2.4, 7.1); arrow(ax, 7.6, 8.0, 7.6, 7.1)
    arrow(ax, 2.4, 5.9, 2.4, 5.1); arrow(ax, 7.6, 5.9, 7.6, 5.1)
    arrow(ax, 2.4, 4.0, 2.4, 3.2)   # s_x -> predictor
    arrow(ax, 2.4, 2.0, 2.4, 1.35)  # predictor -> pred
    ax.add_patch(FancyArrowPatch((3.75, 0.85), (6.4, 4.05), arrowstyle="<|-|>",
                 mutation_scale=12, color=C["green"], lw=2.0,
                 connectionstyle="arc3,rad=-0.25"))
    ax.text(5.1, 2.0, "match in\nREP space", ha="center", fontsize=9, color=C["green"])
    ax.text(7.6, 3.3, "stop-grad", ha="center", fontsize=8.5, color=C["orange"])
    fig.tight_layout()
    save(fig, "jepa_arch_comparison.jpg")


# ---------------------------------------------------------------------------
# 2. I-JEPA masking: one large CONTEXT block predicts several small TARGET
#    blocks -- but the loss is computed on target-encoder REPRESENTATIONS.
# ---------------------------------------------------------------------------
def fig_ijepa_masking():
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # image grid (patchified) on the left
    gx, gy, n, cell = 0.6, 1.0, 6, 0.62
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((gx + j * cell, gy + i * cell), cell, cell,
                         fc="white", ec=C["mgrey"], lw=0.8))
    # context block (big, kept) -- shaded blue
    ax.add_patch(Rectangle((gx + 0 * cell, gy + 0 * cell), 4 * cell, 4 * cell,
                 fc=C["blue"], ec=C["grey"], lw=1.6, alpha=0.35))
    # target blocks (masked, to be predicted) -- green outlines
    tgt = [(4, 4, 2, 2), (0, 4, 2, 1), (4, 1, 1, 2)]
    for (i, j, hh, ww) in tgt:
        ax.add_patch(Rectangle((gx + j * cell, gy + i * cell), ww * cell, hh * cell,
                     fc=C["green"], ec=C["green"], lw=1.8, alpha=0.5))
    ax.text(gx + 3 * cell, gy - 0.5, "image → patches", ha="center", fontsize=10, color=C["text"])
    ax.text(gx + 2 * cell, gy + 2 * cell, "context\n(visible)", ha="center", va="center",
            fontsize=9.5, color=C["text"])
    ax.text(gx + 5 * cell, gy + 5 * cell, "targets\n(masked)", ha="center", va="center",
            fontsize=8.5, color="#3d5a2a")

    # pipeline on the right
    box(ax, 5.6, 4.4, 2.4, 1.0, "encoder\n$E_\\theta$", C["blue"], txt="white", fs=9)
    box(ax, 5.6, 1.0, 2.4, 1.0, "target enc.\n$E_{\\bar\\theta}$ (EMA)", C["orange"], txt="white", fs=8.5)
    box(ax, 8.8, 4.4, 2.4, 1.0, "predictor\n$P_\\phi$\n(+ mask tokens)", C["purple"], txt="white", fs=8.5)
    box(ax, 8.8, 2.7, 2.4, 0.9, "$\\hat s_y$  predicted", C["yellow"], fs=9)
    box(ax, 8.8, 1.05, 2.4, 0.9, "$s_y$  target reps", C["green"], fs=9)

    arrow(ax, 4.5, 4.9, 5.6, 4.9)     # context patches -> encoder
    arrow(ax, 4.5, 1.5, 5.6, 1.5)     # target patches -> target encoder
    arrow(ax, 8.0, 4.9, 8.8, 4.9)     # encoder -> predictor
    arrow(ax, 10.0, 4.4, 10.0, 3.6)   # predictor -> pred reps
    arrow(ax, 8.0, 1.5, 8.8, 1.5)     # target enc -> target reps
    ax.add_patch(FancyArrowPatch((10.0, 2.7), (10.0, 1.95), arrowstyle="<|-|>",
                 mutation_scale=13, color=C["green"], lw=2.0))
    ax.text(11.35, 2.32, "loss in\nrep space", ha="left", va="center", fontsize=8.5, color=C["green"])
    save(fig, "jepa_ijepa_masking.jpg")


# ---------------------------------------------------------------------------
# 3. Representation collapse vs a healthy, spread-out embedding.
# ---------------------------------------------------------------------------
def fig_collapse_energy():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    # left: collapsed
    ax = axes[0]
    ax.set_title("Collapse: encoder ignores input\n(all reps → one point, loss=0 but useless)",
                 fontsize=11, color=C["red"], pad=6)
    pts = np.random.randn(120, 2) * 0.04 + np.array([0.0, 0.0])
    ax.scatter(pts[:, 0], pts[:, 1], s=26, c=C["red"], alpha=0.7, edgecolors="none")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding dim 1"); ax.set_ylabel("embedding dim 2")
    # right: healthy
    ax = axes[1]
    ax.set_title("Healthy: variance kept up, dims decorrelated\n(reps spread across the space)",
                 fontsize=11, color=C["green"], pad=6)
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    pts = np.random.multivariate_normal([0, 0], cov, 200)
    ax.scatter(pts[:, 0], pts[:, 1], s=26, c=C["green"], alpha=0.7, edgecolors="none")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding dim 1"); ax.set_ylabel("embedding dim 2")
    fig.tight_layout()
    save(fig, "jepa_collapse_energy.jpg")


# ---------------------------------------------------------------------------
# 4. VICReg's three terms.
# ---------------------------------------------------------------------------
def fig_vicreg_terms():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    # Variance: histogram of a dimension pushed to have std >= 1
    ax = axes[0]
    ax.set_title("Variance\nkeep each dim's std above a floor", fontsize=11, color=C["text"], pad=6)
    good = np.random.randn(2000) * 1.0
    bad = np.random.randn(2000) * 0.08
    ax.hist(bad, bins=40, color=C["red"], alpha=0.6, label="collapsed (std→0)", density=True)
    ax.hist(good, bins=40, color=C["green"], alpha=0.6, label="healthy (std≥1)", density=True)
    ax.axvline(0, color=C["grey"], lw=0.8)
    ax.set_xlim(-3.5, 3.5); ax.set_yticks([])
    ax.set_xlabel("value of one embedding dim"); ax.legend(fontsize=8, loc="upper right")

    # Invariance: two views' embeddings should coincide
    ax = axes[1]
    ax.set_title("Invariance\ntwo views of the same input match", fontsize=11, color=C["text"], pad=6)
    base = np.random.randn(12, 2) * 1.1
    v2 = base + np.random.randn(12, 2) * 0.12
    ax.scatter(base[:, 0], base[:, 1], s=60, c=C["blue"], label="view x", edgecolors="none")
    ax.scatter(v2[:, 0], v2[:, 1], s=60, c=C["orange"], marker="s", label="view x′", edgecolors="none")
    for a, b in zip(base, v2):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C["grey"], lw=0.7, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("minimize distance between paired embeddings")

    # Covariance: off-diagonal covariance driven to zero
    ax = axes[2]
    ax.set_title("Covariance\ndecorrelate dims (off-diagonals → 0)", fontsize=11, color=C["text"], pad=6)
    M = np.eye(6) + np.random.randn(6, 6) * 0.02
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    im = ax.imshow(np.abs(M), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel("embedding dims"); ax.set_ylabel("embedding dims")
    ax.set_title("Covariance\ndecorrelate dims (off-diagonals → 0)", fontsize=11, color=C["text"], pad=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "jepa_vicreg_terms.jpg")


# ---------------------------------------------------------------------------
# 5. V-JEPA 2-AC world-model planning loop (model-predictive control in latent space).
# ---------------------------------------------------------------------------
def fig_worldmodel_loop():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    box(ax, 0.4, 2.1, 2.3, 1.1, "observation\n(video frame)", C["lgrey"], fs=9)
    box(ax, 3.1, 2.1, 2.1, 1.1, "encoder\n$E_\\theta$", C["blue"], txt="white", fs=9)
    box(ax, 5.6, 2.1, 2.4, 1.1, "state rep\n$s_t$", C["green"], fs=10)
    box(ax, 8.4, 2.1, 2.5, 1.1, "predictor\n$P_\\phi(s_t, a)$", C["purple"], txt="white", fs=9)
    box(ax, 8.4, 0.2, 2.5, 1.0, "predicted\n$\\hat s_{t+1}$", C["yellow"], fs=9.5)
    box(ax, 8.4, 3.9, 2.5, 1.0, "goal rep  $s_{goal}$", C["orange"], txt="white", fs=9)

    arrow(ax, 2.7, 2.65, 3.1, 2.65)
    arrow(ax, 5.2, 2.65, 5.6, 2.65)
    arrow(ax, 8.0, 2.65, 8.4, 2.65)
    arrow(ax, 9.65, 2.1, 9.65, 1.2)      # predictor -> predicted next
    # cost: distance between predicted next and goal
    ax.add_patch(FancyArrowPatch((9.65, 3.9), (9.65, 3.2), arrowstyle="<|-|>",
                 mutation_scale=13, color=C["red"], lw=1.8))
    ax.text(11.05, 3.55, "cost =\n$\\|\\hat s_{t+1}-s_{goal}\\|$", ha="left", va="center",
            fontsize=8.5, color=C["red"])
    # feedback loop: pick action minimizing cost -> back to observation
    ax.add_patch(FancyArrowPatch((8.4, 0.7), (1.55, 2.1), arrowstyle="-|>",
                 mutation_scale=14, color=C["grey"], lw=1.8,
                 connectionstyle="arc3,rad=0.28"))
    ax.text(5.0, 0.35, "search over actions $a$ (MPC), execute best, re-plan",
            ha="center", fontsize=9, color=C["grey"])
    ax.set_title("V-JEPA 2-AC: plan by rolling the predictor forward in LATENT space",
                 fontsize=12, color=C["text"], pad=8)
    save(fig, "jepa_worldmodel_loop.jpg")


if __name__ == "__main__":
    fig_arch_comparison()
    fig_ijepa_masking()
    fig_collapse_energy()
    fig_vicreg_terms()
    fig_worldmodel_loop()
    print("done.")
