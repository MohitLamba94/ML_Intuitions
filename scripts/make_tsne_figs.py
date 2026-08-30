"""Generate figures for misc/tsne.md.

Outputs (all .jpg, per repo convention), written into ../assets:
  tsne_pca_vs_tsne.jpg        -- PCA vs t-SNE on the same labelled toy data
  tsne_gaussian_vs_studentt.jpg -- Gaussian vs Student-t (Cauchy) kernel tails
  tsne_perplexity_sweep.jpg   -- same data embedded at perplexity 5 / 30 / 100
  tsne_springs.jpg            -- attraction/repulsion "spring" schematic
  tsne_pitfalls.jpg           -- misleading cluster size + inter-cluster distance

Run with the repo 'general' conda env:
  /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/make_tsne_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sklearn.datasets import make_blobs, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# Soft, brand-neutral palette (shared with the other figure scripts).
C = {
    "blue":   "#6C8EBF",
    "purple": "#9673A6",
    "green":  "#82B366",
    "orange": "#D6795B",
    "red":    "#C0392B",
    "grey":   "#555555",
    "lgrey":  "#EDEDED",
    "yellow": "#E8B84B",
    "text":   "#222222",
}
CYCLE = [C["blue"], C["orange"], C["green"], C["purple"], C["yellow"]]
DPI = 150
RNG = np.random.RandomState(0)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------------------
# 1. PCA vs t-SNE on the same data.
# Swiss roll: a 2D sheet curled up in 3D. PCA (a flat projection) smears the
# colours together; t-SNE unrolls the neighbourhood order.
# ---------------------------------------------------------------------------
def fig_pca_vs_tsne():
    X, t = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
    # colour by position ALONG the roll (the structure we hope to recover)
    c = t

    pca = PCA(n_components=2).fit_transform(X)
    ts = TSNE(n_components=2, perplexity=40, init="pca",
              random_state=0).fit_transform(X)

    fig = plt.figure(figsize=(11, 3.7))

    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, cmap="Spectral", s=6)
    ax0.set_title("Original data\n(2D sheet curled in 3D)", fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_zticks([])
    ax0.view_init(10, -70)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.scatter(pca[:, 0], pca[:, 1], c=c, cmap="Spectral", s=6)
    ax1.set_title("PCA (linear)\nfar-apart colours overlap", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(ts[:, 0], ts[:, 1], c=c, cmap="Spectral", s=6)
    ax2.set_title("t-SNE (neighbour-based)\nordered strip, colours flow", fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle("Same 1500 points: PCA flattens across the roll, t-SNE follows the neighbourhoods",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "tsne_pca_vs_tsne.jpg")


# ---------------------------------------------------------------------------
# 2. Gaussian vs Student-t tails -- the crowding fix.
# ---------------------------------------------------------------------------
def fig_gaussian_vs_studentt():
    d = np.linspace(0, 6, 400)
    gauss = np.exp(-(d ** 2))          # ~ high-dim p kernel (Gaussian)
    student = 1.0 / (1.0 + d ** 2)     # low-dim q kernel (Student-t, df=1)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(d, gauss, color=C["blue"], lw=2.4, label="Gaussian  (used for $p$, high-dim)")
    ax.plot(d, student, color=C["orange"], lw=2.4, label="Student-t / Cauchy  (used for $q$, 2D map)")

    # highlight the "moderate distance" region where the tails differ most
    ax.axvspan(2.0, 4.0, color=C["lgrey"], alpha=0.7, zorder=0)
    ax.annotate("moderate distances:\nStudent-t keeps weight,\nso points are NOT\nover-attracted",
                xy=(3.0, student[np.argmin(np.abs(d - 3.0))]),
                xytext=(3.4, 0.55), fontsize=9, color=C["text"],
                arrowprops=dict(arrowstyle="->", color=C["grey"]))

    ax.set_xlabel("distance between two points")
    ax.set_ylabel("similarity / affinity")
    ax.set_title("Heavy tail is the 't' in t-SNE: it relieves crowding")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "tsne_gaussian_vs_studentt.jpg")


# ---------------------------------------------------------------------------
# 3. Perplexity sweep on the same blobs.
# ---------------------------------------------------------------------------
def fig_perplexity_sweep():
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=2.6,
                      n_features=10, center_box=(-6, 6), random_state=3)
    perps = [5, 30, 100]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, p in zip(axes, perps):
        emb = TSNE(n_components=2, perplexity=p, init="pca",
                   random_state=0).fit_transform(X)
        for k in range(4):
            m = y == k
            ax.scatter(emb[m, 0], emb[m, 1], s=8, color=CYCLE[k])
        ax.set_title(f"perplexity = {p}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same 4 overlapping clusters (10-D): low perplexity = grainy/local, high = compact & smoothed",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "tsne_perplexity_sweep.jpg")


# ---------------------------------------------------------------------------
# 4. Attraction / repulsion spring schematic.
# ---------------------------------------------------------------------------
def fig_springs():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # a central map point
    p0 = np.array([5.0, 3.0])
    # neighbours that SHOULD be close (high p, currently a bit far) -> pull
    near = np.array([[2.4, 4.4], [3.0, 1.6]])
    # points that should be far (low p, currently too close) -> push
    far = np.array([[6.3, 3.6], [6.1, 2.3]])

    ax.scatter(*p0, s=260, color=C["blue"], zorder=5, edgecolor="white")
    ax.text(p0[0], p0[1] - 0.55, "$y_i$", ha="center", fontsize=12)

    for q in near:
        ax.scatter(*q, s=180, color=C["green"], zorder=5, edgecolor="white")
        arr = FancyArrowPatch(q, q + 0.55 * (p0 - q) / np.linalg.norm(p0 - q),
                              arrowstyle="-|>", mutation_scale=16,
                              color=C["green"], lw=2.2, zorder=4)
        ax.add_patch(arr)
    for q in far:
        ax.scatter(*q, s=180, color=C["red"], zorder=5, edgecolor="white")
        arr = FancyArrowPatch(q, q + 0.7 * (q - p0) / np.linalg.norm(q - p0),
                              arrowstyle="-|>", mutation_scale=16,
                              color=C["red"], lw=2.2, zorder=4)
        ax.add_patch(arr)

    ax.text(2.0, 5.3, "high $p_{ij}$ (true neighbours)\n$\\rightarrow$ ATTRACT (pull in)",
            color=C["green"], fontsize=10, ha="left", weight="bold")
    ax.text(6.6, 5.3, "low $p_{ij}$ but close in map\n$\\rightarrow$ REPEL (push out)",
            color=C["red"], fontsize=10, ha="left", weight="bold")
    ax.set_title("t-SNE gradient = springs: layout settles where pulls balance pushes",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "tsne_springs.jpg")


# ---------------------------------------------------------------------------
# 5. Pitfalls: cluster sizes + inter-cluster distances are not reliable.
# Two clusters with very different spread in the ORIGINAL space come out
# similarly sized in the map; two well-separated groups can sit close.
# ---------------------------------------------------------------------------
def fig_pitfalls():
    # cluster A: tight; cluster B: 5x more spread; cluster C: far away in truth
    a = RNG.randn(250, 5) * 0.5 + np.array([0, 0, 0, 0, 0])
    b = RNG.randn(250, 5) * 2.5 + np.array([12, 0, 0, 0, 0])
    c = RNG.randn(250, 5) * 0.5 + np.array([40, 0, 0, 0, 0])  # very far in truth
    X = np.vstack([a, b, c])
    y = np.array([0] * 250 + [1] * 250 + [2] * 250)

    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=1).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # left: truth via first two dims (roughly), sizes reflect true spread
    for k, lab in zip(range(3), ["A: tight", "B: 5x wider", "C: very far"]):
        m = y == k
        axes[0].scatter(X[m, 0], X[m, 1], s=8, color=CYCLE[k], label=lab)
    axes[0].set_title("Truth (first 2 of 5 dims)\nA tight, B wide, C far off", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for k in range(3):
        m = y == k
        axes[1].scatter(emb[m, 0], emb[m, 1], s=8, color=CYCLE[k])
    axes[1].set_title("t-SNE map\nblobs look same-size; C's true distance is lost",
                      fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle("Do NOT read cluster SIZE or BETWEEN-cluster DISTANCE off a t-SNE plot",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "tsne_pitfalls.jpg")


if __name__ == "__main__":
    fig_pca_vs_tsne()
    fig_gaussian_vs_studentt()
    fig_perplexity_sweep()
    fig_springs()
    fig_pitfalls()
    print("done.")
