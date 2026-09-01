"""
Conceptual figures for `vae.md`.

  Fig 1  vae_ae_vs_vae_latent.jpg  - plain AE latent (clustered, holes, arbitrary
         scale) vs VAE latent (packed against a unit Gaussian). Shows why an AE
         latent is not samplable but a VAE latent is.
  Fig 2  vae_architecture.jpg      - encoder -> (mu, sigma) -> sample z -> decoder
         schematic, with the prior N(0,I) and the two loss terms annotated.
  Fig 3  vae_reparam_trick.jpg     - stochastic-node-inside (can't backprop) vs
         reparameterized (epsilon as external input, gradients flow to mu,sigma).
  Fig 4  vae_prior_sampling_grid.jpg - a 2-D latent grid mapped through a smooth
         synthetic "decoder" so nearby z decode to nearby images: illustrates a
         continuous, walkable latent manifold.
  Fig 5  vqvae_codebook_snap.jpg   - continuous encoder outputs snapped to the
         nearest codebook vector (vector quantization), with Voronoi cells.

Pure numpy / matplotlib (sklearn only for a Voronoi-ish helper we hand-roll).
Deterministic (seeded). Saves .jpg into ../assets.

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/vae_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# dataviz-skill categorical palette (light surface)
BLUE = "#2a78d6"    # slot 1
AQUA = "#1baf7a"    # slot 2
AMBER = "#e0952b"   # slot 3
PLUM = "#8b5cd6"    # slot 4
ROSE = "#d6456b"    # slot 5
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")


def _style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    if grid:
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=9)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ----------------------------------------------------------------------------
# Fig 1 - AE latent (holey) vs VAE latent (packed against N(0,I))
# ----------------------------------------------------------------------------
def fig_ae_vs_vae():
    rng = np.random.default_rng(0)
    n_per = 220
    colors = [BLUE, AQUA, AMBER, PLUM, ROSE]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.3))

    # --- Plain AE: clusters flung to arbitrary, widely separated spots, big gaps
    ae_centers = np.array([[-6.5, 5.0], [7.0, 6.5], [-7.5, -6.0],
                           [5.5, -7.0], [0.5, 0.3]])
    ae_scales = np.array([0.9, 1.3, 0.7, 1.1, 0.8])
    for c, (cen, sc) in enumerate(zip(ae_centers, ae_scales)):
        pts = cen + sc * rng.standard_normal((n_per, 2))
        axA.scatter(pts[:, 0], pts[:, 1], s=10, color=colors[c],
                    alpha=0.7, edgecolors="none", zorder=3)
    # a "random draw" that lands in a hole
    axA.scatter([1.8], [ -3.2], s=140, color=INK, marker="*", zorder=5)
    axA.annotate("draw a random $z$ here\n$\\rightarrow$ decodes to garbage\n(no data ever lived here)",
                 xy=(1.8, -3.2), xytext=(-4.5, -1.2), fontsize=8.5, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    _style(axA)
    axA.set_title("Plain autoencoder latent space", fontsize=12, color=INK, pad=8)
    axA.set_xlabel("$z_1$", fontsize=10); axA.set_ylabel("$z_2$", fontsize=10)
    axA.set_xlim(-11, 11); axA.set_ylim(-11, 11)
    axA.text(0.5, -0.16, "arbitrary scale, isolated clusters, big empty holes",
             transform=axA.transAxes, ha="center", fontsize=9, color=INK2, style="italic")

    # --- VAE: everything packed against a unit Gaussian, overlapping, filling space
    # draw the prior 1- and 2-sigma circles
    for r, a in [(1, 0.10), (2, 0.06)]:
        axB.add_patch(Circle((0, 0), r, fill=True, color=BLUE, alpha=a, zorder=1))
        axB.add_patch(Circle((0, 0), r, fill=False, color=BLUE, lw=1.2, ls="--", zorder=2))
    for c in range(5):
        # each class is a soft blob near the origin, all overlapping -> fills prior
        cen = 0.9 * rng.standard_normal(2)
        pts = cen + 0.75 * rng.standard_normal((n_per, 2))
        axB.scatter(pts[:, 0], pts[:, 1], s=10, color=colors[c],
                    alpha=0.6, edgecolors="none", zorder=3)
    axB.scatter([0.4], [-0.6], s=140, color=INK, marker="*", zorder=5)
    axB.annotate("draw $z\\sim\\mathcal{N}(0,I)$\n$\\rightarrow$ lands on data,\ndecodes to a valid sample",
                 xy=(0.4, -0.6), xytext=(2.1, -2.4), fontsize=8.5, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    _style(axB)
    axB.set_title("VAE latent space", fontsize=12, color=INK, pad=8)
    axB.set_xlabel("$z_1$", fontsize=10); axB.set_ylabel("$z_2$", fontsize=10)
    axB.set_xlim(-4, 4); axB.set_ylim(-4, 4)
    axB.set_aspect("equal")
    axB.text(0.5, -0.16, "KL pulls every code toward the prior $\\rightarrow$ no holes, samplable",
             transform=axB.transAxes, ha="center", fontsize=9, color=INK2, style="italic")

    fig.tight_layout()
    _save(fig, "vae_ae_vs_vae_latent.jpg")


# ----------------------------------------------------------------------------
# Fig 2 - VAE architecture schematic
# ----------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec=INK2, fs=10, tc=INK):
    b = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.4, zorder=3)
    ax.add_patch(b)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4)


def _arrow(ax, p0, p1, color=INK, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=2,
                                 shrinkA=2, shrinkB=2))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 5.0); ax.axis("off")

    y = 2.6
    _box(ax, (1.15, y), 1.5, 1.5, "input\n$x$", "#eef3fb", ec=BLUE)
    _box(ax, (3.35, y), 1.7, 1.9, "encoder\n$q_\\phi(z\\mid x)$", "#eaf1fb", ec=BLUE, fs=10)
    # mu / sigma heads
    _box(ax, (5.5, 3.55), 1.2, 0.9, "$\\mu_\\phi(x)$", "#fff4e6", ec=AMBER, fs=11)
    _box(ax, (5.5, 1.65), 1.2, 0.9, "$\\sigma_\\phi(x)$", "#fff4e6", ec=AMBER, fs=11)
    _box(ax, (7.5, y), 1.35, 1.2, "sample\n$z$", "#ecf7f1", ec=AQUA, fs=10)
    _box(ax, (9.7, y), 1.7, 1.9, "decoder\n$p_\\theta(x\\mid z)$", "#f3ecfb", ec=PLUM, fs=10)
    _box(ax, (11.5, y), 1.2, 1.5, "$\\hat{x}$", "#f3ecfb", ec=PLUM, fs=12)

    _arrow(ax, (1.9, y), (2.5, y))
    _arrow(ax, (4.2, y + 0.35), (4.9, 3.35), color=AMBER)
    _arrow(ax, (4.2, y - 0.35), (4.9, 1.75), color=AMBER)
    _arrow(ax, (6.1, 3.35), (6.85, y + 0.25), color=AQUA)
    _arrow(ax, (6.1, 1.75), (6.85, y - 0.25), color=AQUA)
    _arrow(ax, (8.2, y), (8.85, y))
    _arrow(ax, (10.55, y), (10.9, y))

    # epsilon injection into the sample node
    ax.text(7.5, 4.35, "$\\epsilon\\sim\\mathcal{N}(0,I)$", ha="center", fontsize=10, color=INK2)
    _arrow(ax, (7.5, 4.05), (7.5, 3.25), color=INK2, lw=1.4)
    ax.text(8.55, 3.35, "$z=\\mu+\\sigma\\odot\\epsilon$", ha="left", fontsize=9.5,
            color=INK2, style="italic")

    # loss annotations
    ax.text(9.7, 0.55, "reconstruction term\n$\\mathbb{E}_{q_\\phi}[\\log p_\\theta(x\\mid z)]$",
            ha="center", fontsize=9, color=PLUM)
    ax.text(4.4, 0.55, "KL term  $\\mathrm{KL}\\!\\left(q_\\phi(z\\mid x)\\,\\|\\,p(z)\\right)$\npulls $q_\\phi$ toward prior $\\mathcal{N}(0,I)$",
            ha="center", fontsize=9, color=BLUE)

    ax.set_title("Variational Autoencoder: encode to a distribution, sample, decode",
                 fontsize=12.5, color=INK, pad=6)
    _save(fig, "vae_architecture.jpg")


# ----------------------------------------------------------------------------
# Fig 3 - reparameterization trick
# ----------------------------------------------------------------------------
def fig_reparam():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax in (axL, axR):
        ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")

    # LEFT: stochastic node in the path (blocks gradient)
    axL.set_title("Naive: sample inside the graph", fontsize=12, color=INK, pad=6)
    _box(axL, (1.0, 2.5), 1.2, 1.1, "$\\phi$", "#fff4e6", ec=AMBER, fs=13)
    _box(axL, (3.0, 2.5), 1.7, 1.3, "$z\\sim\\mathcal{N}(\\mu_\\phi,\\sigma_\\phi^2)$",
         "#fdeaea", ec=ROSE, fs=10)
    _box(axL, (5.1, 2.5), 1.2, 1.1, "loss", "#f3ecfb", ec=PLUM, fs=11)
    _arrow(axL, (1.65, 2.5), (2.1, 2.5))
    _arrow(axL, (3.9, 2.5), (4.45, 2.5))
    # blocked backward arrow
    axL.add_patch(FancyArrowPatch((4.45, 1.9), (3.9, 1.9), arrowstyle="-|>",
                  mutation_scale=15, color=ROSE, lw=1.8, ls=(0, (4, 3))))
    axL.text(3.0, 1.15, "random node has no\nusable gradient $\\partial z/\\partial\\phi$",
             ha="center", fontsize=9, color=ROSE)
    axL.plot(3.0, 1.72, marker="x", ms=13, color=ROSE, mew=3)

    # RIGHT: reparameterized
    axR.set_title("Reparameterized: randomness as an input", fontsize=12, color=INK, pad=6)
    _box(axR, (1.0, 3.4), 1.2, 1.0, "$\\phi$", "#fff4e6", ec=AMBER, fs=13)
    _box(axR, (1.0, 1.2), 1.4, 1.0, "$\\epsilon\\sim\\mathcal{N}(0,I)$", "#eef3fb", ec=BLUE, fs=9.5)
    _box(axR, (3.15, 2.5), 1.9, 1.25, "$z=\\mu_\\phi+\\sigma_\\phi\\odot\\epsilon$",
         "#ecf7f1", ec=AQUA, fs=10)
    _box(axR, (5.2, 2.5), 1.1, 1.0, "loss", "#f3ecfb", ec=PLUM, fs=11)
    _arrow(axR, (1.65, 3.3), (2.25, 2.85))
    _arrow(axR, (1.75, 1.35), (2.25, 2.05), color=BLUE)
    _arrow(axR, (4.15, 2.5), (4.6, 2.5))
    # clean backward gradient
    axR.add_patch(FancyArrowPatch((4.6, 1.75), (2.35, 1.75), arrowstyle="-|>",
                  mutation_scale=15, color=AQUA, lw=2.0))
    axR.text(3.7, 0.95, "deterministic in $\\phi$ given $\\epsilon$\n$\\rightarrow$ gradient flows to $\\phi$",
             ha="center", fontsize=9, color=AQUA)

    _save(fig, "vae_reparam_trick.jpg")


# ----------------------------------------------------------------------------
# Fig 4 - smooth latent manifold: a 2-D grid of z decoded by a synthetic decoder
# ----------------------------------------------------------------------------
def _synthetic_decode(z1, z2):
    """A smooth map from 2-D latent to a little 12x12 'image'. Nearby z -> nearby
    image, so the grid shows a continuously morphing shape (a generic blob whose
    orientation/size/curvature vary smoothly with z)."""
    n = 12
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs)
    ang = 0.9 * z1
    Xr = X * np.cos(ang) - Y * np.sin(ang)
    Yr = X * np.sin(ang) + Y * np.cos(ang)
    a = 0.55 + 0.22 * z2         # aspect ratio varies with z2
    b = 0.55 - 0.18 * z2
    r2 = (Xr / a) ** 2 + (Yr / b) ** 2
    img = np.exp(-2.5 * r2)
    img += 0.25 * np.exp(-6 * ((Xr - 0.35 * z1) ** 2 + (Yr) ** 2))
    return img


def fig_manifold_grid():
    g = 9
    zs = np.linspace(-2, 2, g)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    canvas = np.zeros((g * 12, g * 12))
    for i, z2 in enumerate(zs[::-1]):      # top = high z2
        for j, z1 in enumerate(zs):
            canvas[i * 12:(i + 1) * 12, j * 12:(j + 1) * 12] = _synthetic_decode(z1, z2)
    ax.imshow(canvas, cmap="magma", origin="upper", extent=[-2.4, 2.4, -2.4, 2.4])
    ax.set_xlabel("$z_1$", fontsize=11, color=INK)
    ax.set_ylabel("$z_2$", fontsize=11, color=INK)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("Decoding a grid of latent codes $z$\n"
                 "nearby codes decode to nearby outputs (a smooth, walkable manifold)",
                 fontsize=11.5, color=INK, pad=8)
    _save(fig, "vae_prior_sampling_grid.jpg")


# ----------------------------------------------------------------------------
# Fig 5 - VQ-VAE codebook snapping (vector quantization)
# ----------------------------------------------------------------------------
def fig_vqvae():
    rng = np.random.default_rng(3)
    # codebook vectors
    K = 7
    codes = np.array([[-2.2, 1.6], [-0.4, 2.3], [1.9, 1.9], [2.6, -0.3],
                      [1.1, -2.0], [-1.3, -1.7], [-2.6, -0.2]])
    # continuous encoder outputs scattered around
    pts = rng.uniform(-3.2, 3.2, size=(160, 2))

    # assign each point to nearest code
    d = ((pts[:, None, :] - codes[None, :, :]) ** 2).sum(-1)
    assign = d.argmin(1)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    # Voronoi cells via fine background grid coloring
    gx = np.linspace(-3.4, 3.4, 400)
    GX, GY = np.meshgrid(gx, gx)
    GP = np.stack([GX.ravel(), GY.ravel()], 1)
    gd = ((GP[:, None, :] - codes[None, :, :]) ** 2).sum(-1).argmin(1).reshape(GX.shape)
    cmap = matplotlib.colors.ListedColormap(
        ["#eef3fb", "#ecf7f1", "#fff4e6", "#f3ecfb", "#fdeaea", "#e9f7f9", "#f2f0e8"])
    ax.imshow(gd, extent=[-3.4, 3.4, -3.4, 3.4], origin="lower",
              cmap=cmap, alpha=0.65, zorder=0, aspect="auto")

    palette = [BLUE, AQUA, AMBER, PLUM, ROSE, "#1f9bb0", INK2]
    for k in range(K):
        sel = assign == k
        # snap lines
        for p in pts[sel]:
            ax.plot([p[0], codes[k, 0]], [p[1], codes[k, 1]],
                    color=palette[k], lw=0.5, alpha=0.4, zorder=2)
        ax.scatter(pts[sel, 0], pts[sel, 1], s=14, color=palette[k],
                   alpha=0.55, edgecolors="none", zorder=3)
    ax.scatter(codes[:, 0], codes[:, 1], s=240, marker="*", color=INK,
               edgecolors="white", linewidths=1.4, zorder=5)
    for k in range(K):
        ax.text(codes[k, 0], codes[k, 1] + 0.28, f"$e_{k+1}$", ha="center",
                fontsize=10, color=INK, zorder=6)

    _style(ax, grid=False)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
    ax.set_xlabel("encoder output dim 1", fontsize=10)
    ax.set_ylabel("encoder output dim 2", fontsize=10)
    ax.set_title("VQ-VAE: every continuous encoder output is snapped\n"
                 "to its nearest codebook vector $e_k$ (vector quantization)",
                 fontsize=11.5, color=INK, pad=8)
    _save(fig, "vqvae_codebook_snap.jpg")


if __name__ == "__main__":
    fig_ae_vs_vae()
    fig_architecture()
    fig_reparam()
    fig_manifold_grid()
    fig_vqvae()
    print("done.")
