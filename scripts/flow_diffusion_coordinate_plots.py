"""
Plots for `ddpm_ddim_flow_score.md`, section 4.7 ("Two sides of the same coin").

Visualizes why DDPM (variance-preserving) and flow matching (straight-line) are
the SAME trajectory in different coordinates:

  Fig 1  schedule as a curve in the (alpha, sigma) plane; the per-time radial
         rescaling c_t that maps the straight chord onto the VP arc, and how
         normalizing the scale collapses both onto one arc.
  Fig 2  the payoff: for a fixed (x, epsilon) pair, z_t = alpha_t x + sigma_t eps
         is a STRAIGHT segment for flow matching and a CURVED path for VP,
         between the very same endpoints.
  Fig 3  SNR is the shared clock: two schedules tracing the same straight chord
         but on different time clocks pass SNR = 1 at different t (a pure
         time-reparameterization).

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python \
        scripts/flow_diffusion_coordinate_plots.py

Saves JPGs into ../assets/ relative to this file.
Colors follow the dataviz skill categorical palette (slot 1 blue, slot 2 aqua).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (dataviz skill: categorical slots on the light surface) ----
FM = "#2a78d6"     # flow matching / straight  (categorical slot 1, blue)
VP = "#1baf7a"     # diffusion / VP  (categorical slot 2, aqua)
INK = "#0b0b0b"    # text-primary
INK2 = "#52514e"   # text-secondary
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ---------------------------------------------------------------- Fig 1
def fig_alpha_sigma_plane():
    t = np.linspace(0, 1, 200)
    # flow matching: straight chord (1-t, t)
    a_fm, s_fm = 1 - t, t
    # VP: unit quarter-circle arc
    a_vp, s_vp = np.cos(np.pi * t / 2), np.sin(np.pi * t / 2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5))

    # ---- left: raw coordinates + radial rescaling ----
    _style(axL)
    axL.plot(a_fm, s_fm, color=FM, lw=2.2, label="flow matching  (1−t, t)")
    axL.plot(a_vp, s_vp, color=VP, lw=2.2, label="diffusion / VP  (cos, sin)")
    for tm in (0.25, 0.5, 0.75):
        p = np.array([1 - tm, tm])          # chord point
        n = p / np.linalg.norm(p)            # its unit-norm projection (on arc)
        axL.plot([0, n[0]], [0, n[1]], color=INK2, lw=0.9, ls="--", alpha=0.6, zorder=1)
        axL.scatter(*p, color=FM, s=40, zorder=3)
        axL.scatter(*n, facecolors="none", edgecolors=VP, s=55, lw=1.6, zorder=3)
    axL.annotate("radial rescale $c_t$", xy=(0.354, 0.354),
                 xytext=(0.52, 0.16), color=INK2, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    axL.scatter([1], [0], color=INK, s=25, zorder=4)
    axL.scatter([0], [1], color=INK, s=25, zorder=4)
    axL.annotate("data  (1, 0)", (1, 0), xytext=(0.62, -0.02), color=INK, fontsize=9)
    axL.annotate("noise  (0, 1)", (0, 1), xytext=(0.03, 1.0), color=INK, fontsize=9)
    axL.set_title("Raw coordinates: the schedule is a curve in the $(\\alpha,\\sigma)$ plane",
                  color=INK, fontsize=10.5)
    axL.set_xlabel("signal coefficient  $\\alpha_t$", color=INK2)
    axL.set_ylabel("noise coefficient  $\\sigma_t$", color=INK2)
    axL.set_aspect("equal")
    axL.set_xlim(-0.05, 1.15); axL.set_ylim(-0.1, 1.15)
    axL.legend(loc="upper right", fontsize=8.5, frameon=False)

    # ---- right: normalize the radial scale -> same arc ----
    _style(axR)
    norm_fm = np.stack([a_fm, s_fm]) / np.linalg.norm(np.stack([a_fm, s_fm]), axis=0)
    axR.plot(norm_fm[0], norm_fm[1], color=FM, lw=3.4,
             label="flow matching, normalized")
    axR.plot(a_vp, s_vp, color=VP, lw=1.8, ls="--",
             label="diffusion / VP")
    axR.annotate("both lie on the same\nunit quarter-arc", xy=(0.707, 0.707),
                 xytext=(0.15, 0.35), color=INK2, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    axR.set_title("Normalize the radial scale $\\to$ identical arc",
                  color=INK, fontsize=10.5)
    axR.set_xlabel("normalized $\\alpha_t$", color=INK2)
    axR.set_ylabel("normalized $\\sigma_t$", color=INK2)
    axR.set_aspect("equal")
    axR.set_xlim(-0.05, 1.15); axR.set_ylim(-0.05, 1.15)
    axR.legend(loc="upper right", fontsize=8.5, frameon=False)

    _save(fig, "schedule_alpha_sigma_plane.jpg")


# ---------------------------------------------------------------- Fig 2
def fig_curved_vs_straight():
    x = np.array([1.6, 0.5])     # a clean data point
    eps = np.array([-0.7, 1.4])  # a noise point
    t = np.linspace(0, 1, 200)
    tm = np.linspace(0, 1, 7)    # marker times

    def path(a_fn, s_fn, tt):
        a, s = a_fn(tt), s_fn(tt)
        return np.outer(a, x) + np.outer(s, eps)

    fm = path(lambda u: 1 - u, lambda u: u, t)
    vp = path(lambda u: np.cos(np.pi * u / 2), lambda u: np.sin(np.pi * u / 2), t)
    fm_m = path(lambda u: 1 - u, lambda u: u, tm)
    vp_m = path(lambda u: np.cos(np.pi * u / 2), lambda u: np.sin(np.pi * u / 2), tm)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _style(ax)
    ax.plot(fm[:, 0], fm[:, 1], color=FM, lw=2.2, label="flow matching  (straight)")
    ax.plot(vp[:, 0], vp[:, 1], color=VP, lw=2.2, label="diffusion / VP  (curved)")
    ax.scatter(fm_m[:, 0], fm_m[:, 1], color=FM, s=28, zorder=3)
    ax.scatter(vp_m[:, 0], vp_m[:, 1], color=VP, s=28, zorder=3)
    ax.scatter(*x, color=INK, s=55, zorder=4)
    ax.scatter(*eps, color=INK, s=55, zorder=4)
    ax.annotate("$x$  (clean data, $t=0$)", x, xytext=(x[0] - 0.15, x[1] - 0.28),
                color=INK, fontsize=9.5)
    ax.annotate("$\\epsilon$  (noise, $t=1$)", eps, xytext=(eps[0] - 0.15, eps[1] + 0.12),
                color=INK, fontsize=9.5)
    ax.set_title("Same endpoints $x,\\ \\epsilon$ — only the schedule bends the path\n"
                 "$z_t = \\alpha_t x + \\sigma_t \\epsilon$", color=INK, fontsize=10.5)
    ax.set_xlabel("dimension 1", color=INK2)
    ax.set_ylabel("dimension 2", color=INK2)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    _save(fig, "curved_vs_straight_paths.jpg")


# ---------------------------------------------------------------- Fig 3
def fig_snr_clock():
    t = np.linspace(0.02, 0.98, 400)
    # both schedules trace the SAME straight chord (1-u, u); only the clock differs
    logsnr = lambda u: 2 * np.log((1 - u) / u)      # log SNR = 2 log(alpha/sigma)
    fm = logsnr(t)          # u = t
    warp = logsnr(t ** 2)   # u = t^2  (time-warped: same path, different clock)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    _style(ax)
    ax.plot(t, fm, color=FM, lw=2.2, label="flow clock  ($u=t$)")
    ax.plot(t, warp, color=VP, lw=2.2, label="warped clock  ($u=t^2$)")
    ax.axhline(0, color=INK2, lw=1.0, ls=":")
    ax.text(0.985, 0.15, "SNR = 1", color=INK2, fontsize=9, ha="right")
    # crossings of log SNR = 0  ->  u = 0.5
    ax.scatter([0.5], [0], color=FM, s=45, zorder=4)
    ax.scatter([np.sqrt(0.5)], [0], color=VP, s=45, zorder=4)
    ax.annotate("$t=0.5$", (0.5, 0), xytext=(0.42, 1.6), color=FM, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=FM, lw=0.9))
    ax.annotate("$t\\approx0.707$", (np.sqrt(0.5), 0), xytext=(0.72, -3.2), color=VP,
                fontsize=9, arrowprops=dict(arrowstyle="->", color=VP, lw=0.9))
    ax.annotate("same state (SNR=1),\ndifferent clock reading", xy=(0.6, 0),
                xytext=(0.30, -5.6), color=INK2, fontsize=9)
    ax.set_title("SNR is the shared clock; a schedule chooses\n"
                 "when to pass each SNR (a time-reparameterization)",
                 color=INK, fontsize=10.5)
    ax.set_xlabel("time  $t$", color=INK2)
    ax.set_ylabel("log-SNR$(t) = 2\\log(\\alpha_t/\\sigma_t)$", color=INK2)
    ax.set_ylim(-8, 8)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    _save(fig, "snr_clock.jpg")


if __name__ == "__main__":
    fig_alpha_sigma_plane()
    fig_curved_vs_straight()
    fig_snr_clock()
    print("done")
