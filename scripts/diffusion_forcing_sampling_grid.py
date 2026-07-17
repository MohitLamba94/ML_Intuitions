"""
Plot for `diffusion-forcing.md` — the 2D sampling grid.

Diffusion Forcing samples on a grid whose columns are sequence positions
(tokens / frames) and whose rows are denoising iterations. Each cell holds the
CURRENT noise level of that token; generation is a schedule that lowers every
column from K (pure noise, dark) down to 0 (clean, light). Different schedules
= different paths through the same grid, all reachable from ONE trained model:

  Full-sequence diffusion   every column denoised together  (uniform rows)
  Autoregressive / next-tok denoise one token at a time     (staircase front)
  Diffusion Forcing pyramid future kept noisier than present (diagonal band)

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python \
        scripts/diffusion_forcing_sampling_grid.py

Saves the JPG into ../assets/ relative to this file.
Sequential (single-hue) ramp: dark = high noise, light = clean, per the dataviz
skill's sequential-palette guidance.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---- palette ----
INK = "#0b0b0b"     # text-primary
INK2 = "#52514e"    # text-secondary
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"
# sequential single-hue ramp (light surface -> deep blue), clean -> noisy
CMAP = LinearSegmentedColormap.from_list(
    "noise", ["#f4f8fd", "#9ec4ec", "#2a78d6", "#123a6b"]
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

T = 8      # sequence positions (columns)
M = 8      # denoising iterations (rows), row 0 = start (all noise)


def _grids():
    """Return three (M x T) noise-level arrays in [0,1] (1 = pure noise)."""
    pos = np.arange(T) / (T - 1)                 # 0..1 across positions
    p = np.arange(M) / (M - 1)                   # 0..1 progress down the rows

    full = np.zeros((M, T))
    auto = np.zeros((M, T))
    pyr = np.zeros((M, T))

    for m in range(M):
        # full-sequence: uniform level across all columns, lowered each row
        full[m, :] = 1.0 - p[m]

        # autoregressive: a sharp front sweeps left->right; token clean once passed
        front = p[m] * (T - 1)
        auto[m, :] = np.clip((np.arange(T) - front) / 0.9 + 0.5, 0, 1)

        # pyramid: within a row noise rises with position (future noisier),
        # and the whole ramp is lowered over rows until it hits 0
        pyr[m, :] = np.clip(1.0 - 2.0 * p[m] + pos, 0, 1)

    return full, auto, pyr


def _panel(ax, grid, title):
    im = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, color=INK, fontsize=10.5, pad=8)
    ax.set_xlabel("sequence position  $t\\;\\rightarrow$", color=INK2, fontsize=9)
    ax.set_xticks(range(T))
    ax.set_xticklabels([str(i + 1) for i in range(T)], fontsize=8, color=INK2)
    ax.set_yticks(range(M))
    ax.set_yticklabels([str(i) for i in range(M)], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    # thin white gridlines between cells
    ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    return im


def fig_sampling_grid():
    full, auto, pyr = _grids()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))

    _panel(axes[0], full, "Full-sequence diffusion\n(all columns denoise together)")
    _panel(axes[1], auto, "Next-token / autoregressive\n(one token at a time)")
    im = _panel(axes[2], pyr, "Diffusion Forcing: pyramid\n(future kept noisier)")

    axes[0].set_ylabel("denoising iteration\n(top: all noise $\\to$ bottom: clean)",
                       color=INK2, fontsize=9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("noise level of token   ($K$ = pure noise, $0$ = clean)",
                   color=INK2, fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["$0$", "$K$"])
    cbar.ax.tick_params(colors=INK2)

    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    path = os.path.join(ASSETS, "diffusion_forcing_sampling_grid.jpg")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    fig_sampling_grid()
    print("done")
