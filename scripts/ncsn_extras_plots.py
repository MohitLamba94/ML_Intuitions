"""
Figures for the "Extras" section of `ddpm_ddim_flow_score.md` (NCSN / score matching
across noise scales; Song & Ermon 2019).  Recreated in our notation, matplotlib only.

  Fig A  the pitfall: the score is reliable near data (high density) but unreliable
         in the low-density gap between modes, so a single-scale walk gets stuck.
  Fig B  the fix: a ladder of Gaussian noise scales fills in the empty space
         (big noise) while keeping detail (small noise) -> score learnable everywhere.
  Fig C  annealed Langevin: sample at the largest noise scale, then walk down the
         ladder, so coarse structure forms first and detail last.

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/ncsn_extras_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

# two well-separated modes -> a genuine low-density gap between them
MU = np.array([[-1.5, 0.0], [1.5, 0.0]])
W = np.array([0.5, 0.5])


def comp_pdf(X, mu, sig):
    d2 = ((X - mu) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sig ** 2)) / (2 * np.pi * sig ** 2)


def density(X, sig):
    return sum(w * comp_pdf(X, m, sig) for w, m in zip(W, MU))


def score(X, sig):
    num = np.zeros_like(X); den = np.zeros(X.shape[:-1])
    for w, m in zip(W, MU):
        pk = w * comp_pdf(X, m, sig)
        den += pk; num += pk[..., None] * (m - X) / sig ** 2
    return num / (den[..., None] + 1e-12)


def _ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.4, 2.4)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    p = os.path.join(ASSETS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(p))


def sample_data(n, sig, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 2, n)
    return MU[idx] + rng.normal(0, sig, (n, 2))


# ---------------------------------------------------------------- Fig A: pitfall
def fig_pitfall():
    sig = 0.35
    g = np.linspace(-3.2, 3.2, 240); gy = np.linspace(-2.4, 2.4, 200)
    gx, gyy = np.meshgrid(g, gy)
    grid = np.stack([gx, gyy], -1)
    dens = density(grid, sig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _ax(ax)
    ax.contourf(gx, gyy, dens, levels=16, cmap="Blues", alpha=0.9)
    q = np.linspace(-3.0, 3.0, 19); qy = np.linspace(-2.1, 2.1, 11)
    qx, qyy = np.meshgrid(q, qy)
    qpts = np.stack([qx, qyy], -1)
    sv = score(qpts, sig)
    n = np.linalg.norm(sv, axis=-1, keepdims=True)
    svn = sv / (n + 1e-9)
    # shade arrows by data density: low-density = unreliable (light/red-ish gray)
    dq = density(qpts, sig)
    rel = (dq > 0.02)
    ax.quiver(qx[rel], qyy[rel], svn[..., 0][rel], svn[..., 1][rel],
              color=INK2, alpha=0.9, scale=26, width=0.005)
    ax.quiver(qx[~rel], qyy[~rel], svn[..., 0][~rel], svn[..., 1][~rel],
              color="#c98500", alpha=0.6, scale=26, width=0.005)
    ax.axvspan(-0.7, 0.7, color="#eda100", alpha=0.10)
    ax.text(0.0, 1.9, "low-density gap:\nscore barely learned here", color="#8a5a00",
            fontsize=8.5, ha="center", va="top")
    ax.set_title("The pitfall: score is trustworthy near data, shaky in the empty gap",
                 color=INK, fontsize=10.5)
    _save(fig, "ncsn_pitfall.jpg")


# ---------------------------------------------------------------- Fig B & C combined
def fig_ladder_and_anneal():
    scales = [1.4, 0.7, 0.35]           # noise ladder: large -> small
    g = np.linspace(-3.2, 3.2, 240); gy = np.linspace(-2.4, 2.4, 200)
    gx, gyy = np.meshgrid(g, gy)
    grid = np.stack([gx, gyy], -1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))

    # top row (Fig B): the perturbed densities at each scale
    for ax, sig in zip(axes[0], scales):
        _ax(ax)
        ax.contourf(gx, gyy, density(grid, sig), levels=16, cmap="Blues", alpha=0.9)
        pts = sample_data(400, sig, seed=int(sig * 100))
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=INK2, alpha=0.35, linewidths=0)
        ax.set_title(f"noise scale $\\sigma={sig}$", color=INK, fontsize=10)
    axes[0, 0].text(-3.0, 2.1, "large noise:\ngap filled in", color=INK2, fontsize=8.5, va="top")
    axes[0, 2].text(-3.0, 2.1, "small noise:\ndetail kept", color=INK2, fontsize=8.5, va="top")

    # bottom row (Fig C): annealed Langevin walking down the ladder
    rng = np.random.default_rng(7)
    n_walk = 120
    x = rng.uniform([-3, -2], [3, 2], size=(n_walk, 2))   # start spread out
    for ax, sig in zip(axes[1], scales):
        _ax(ax)
        ax.contourf(gx, gyy, density(grid, sig), levels=16, cmap="Blues", alpha=0.35)
        eta = 0.08 * sig ** 2                              # step scaled to noise level
        for _ in range(60):
            x = x + eta * score(x, sig) + np.sqrt(2 * eta) * rng.normal(0, 1, x.shape)
        ax.scatter(x[:, 0], x[:, 1], s=7, color=AQUA, alpha=0.7, linewidths=0)
        ax.set_title(f"anneal at $\\sigma={sig}$", color=INK, fontsize=10)
    axes[1, 0].text(-3.0, 2.1, "samples start\neverywhere", color=INK2, fontsize=8.5, va="top")
    axes[1, 2].text(-3.0, 2.1, "collapse onto\nthe two modes", color=INK2, fontsize=8.5, va="top")

    fig.text(0.5, 0.99, "Fix: perturb data at many noise scales (top), then anneal Langevin down the ladder (bottom)",
             ha="center", color=INK, fontsize=11)
    _save(fig, "ncsn_ladder_anneal.jpg")


if __name__ == "__main__":
    fig_pitfall()
    fig_ladder_and_anneal()
    print("done")
