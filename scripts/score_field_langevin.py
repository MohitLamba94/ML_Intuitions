"""
Score-field + Langevin figure for `ddpm_ddim_flow_score.md`, section 4.3.

Recreates, in our own notation, the intuition of Yang Song's score blog
(score_contour + langevin), with matplotlib only (no external assets):

  Left   the score field  s(x) = grad_x log p(x)  as arrows over a density,
         pointing "uphill" toward higher-probability regions.
  Right  Langevin sampling  x <- x + eta * s(x) + sqrt(2 eta) * z  from
         scattered starts, migrating to the modes.

The density is a fixed 2-D Gaussian mixture, whose score is closed-form
(no training needed).

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/score_field_langevin.py
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

# fixed Gaussian mixture: means, shared isotropic sigma, equal weights
MU = np.array([[-1.2, 0.8], [1.3, 0.6], [0.0, -1.2]])
SIG = 0.5
W = np.array([1 / 3, 1 / 3, 1 / 3])


def _comp_pdf(X, mu):
    d2 = ((X - mu) ** 2).sum(-1)
    return np.exp(-d2 / (2 * SIG ** 2)) / (2 * np.pi * SIG ** 2)


def density(X):
    return sum(w * _comp_pdf(X, m) for w, m in zip(W, MU))


def score(X):
    # grad_x log p(x) = [ sum_k w_k N_k(x) * (mu_k - x)/sigma^2 ] / [ sum_k w_k N_k(x) ]
    num = np.zeros_like(X)
    den = np.zeros(X.shape[:-1])
    for w, m in zip(W, MU):
        pk = w * _comp_pdf(X, m)
        den += pk
        num += pk[..., None] * (m - X) / SIG ** 2
    return num / (den[..., None] + 1e-12)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)


def main():
    g = np.linspace(-3, 3, 300)
    gx, gy = np.meshgrid(g, g)
    grid = np.stack([gx, gy], -1)
    dens = density(grid)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---- left: score field over the density ----
    _style(axL)
    axL.contourf(gx, gy, dens, levels=18, cmap="Blues", alpha=0.9)
    q = np.linspace(-2.6, 2.6, 17)
    qx, qy = np.meshgrid(q, q)
    qpts = np.stack([qx, qy], -1)
    sv = score(qpts)
    n = np.linalg.norm(sv, axis=-1, keepdims=True)
    svn = sv / (n + 1e-9)                       # normalize arrows for readability
    axL.quiver(qx, qy, svn[..., 0], svn[..., 1], color=INK2, alpha=0.75,
               scale=32, width=0.004)
    axL.scatter(MU[:, 0], MU[:, 1], color=INK, s=18, zorder=5)
    axL.set_title("The score  $s(x)=\\nabla_x\\log p(x)$\narrows point uphill toward higher density",
                  color=INK, fontsize=10.5)

    # ---- right: Langevin sampling trajectories ----
    _style(axR)
    axR.contourf(gx, gy, dens, levels=18, cmap="Blues", alpha=0.45)
    rng = np.random.default_rng(3)
    starts = rng.uniform(-2.6, 2.6, size=(9, 2))
    eta = 0.012
    steps = 220
    for s0 in starts:
        x = s0.copy(); traj = [x.copy()]
        for _ in range(steps):
            x = x + eta * score(x[None, :])[0] + np.sqrt(2 * eta) * rng.normal(0, 1, 2)
            traj.append(x.copy())
        traj = np.array(traj)
        axR.plot(traj[:, 0], traj[:, 1], color=AQUA, lw=1.0, alpha=0.85, zorder=2)
        axR.scatter(*s0, facecolors="none", edgecolors=INK2, s=34, lw=1.3, zorder=3)
        axR.scatter(*traj[-1], color=AQUA, s=26, zorder=4)
    axR.scatter(MU[:, 0], MU[:, 1], color=INK, s=18, zorder=5)
    axR.set_title("Langevin sampling: follow the score ($+$ noise)\nfrom random starts (○) to the data modes",
                  color=INK, fontsize=10.5)

    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(ASSETS, "score_field_langevin.jpg")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
