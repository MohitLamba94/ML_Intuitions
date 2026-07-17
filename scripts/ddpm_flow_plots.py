"""
Toy-example figures for `ddpm_ddim_flow_score.md`, Parts 1 (DDPM) and 2 (Rectified Flow).

  Fig 1  forward corruption + reverse generation of a 2-D two-moons dataset
         (the "destroy then rebuild" picture).                     -> Part 1
  Fig 2  the noise schedule: beta_t, alpha-bar_t, and the signal vs noise
         coefficients sqrt(alpha-bar_t), sqrt(1-alpha-bar_t).       -> Part 1
  Fig 3  random straight-line pairings CROSS; the learned marginal (rectified)
         flow reroutes into non-crossing, gently curved paths.      -> Part 2
  Fig 4  a smooth deterministic ODE path vs several jittery stochastic SDE
         paths from the same start.                                 -> Part 2

Pure numpy/matplotlib (no sklearn/torch). Deterministic (seeded).

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/ddpm_flow_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# dataviz-skill categorical palette (light surface)
BLUE = "#2a78d6"   # slot 1
AQUA = "#1baf7a"   # slot 2
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
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def two_moons(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n // 2)
    outer = np.stack([np.cos(t), np.sin(t)], 1)
    inner = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    pts = np.concatenate([outer, inner], 0)
    pts = (pts - pts.mean(0)) / pts.std(0)          # normalize to ~unit variance
    pts += rng.normal(0, 0.06, pts.shape)
    return pts


# ---------------------------------------------------------------- Fig 1
def fig_forward_reverse():
    x0 = two_moons(600)
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 1, x0.shape)                # one fixed noise realization
    abars = [1.0, 0.7, 0.4, 0.15, 0.0]              # signal power alpha-bar_t

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, ab in zip(axes, abars):
        z = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        ax.scatter(z[:, 0], z[:, 1], s=5, color=BLUE, alpha=0.55, linewidths=0)
        _style(ax, grid=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"$\\bar\\alpha_t={ab:.2f}$", color=INK, fontsize=10)
    axes[0].set_title("$\\bar\\alpha_t=1.00$  (data)", color=INK, fontsize=10)
    axes[-1].set_title("$\\bar\\alpha_t=0.00$  (noise)", color=INK, fontsize=10)
    fig.text(0.5, 1.02, "forward (fixed):  add noise, data $\\rightarrow$ noise",
             ha="center", color=BLUE, fontsize=11)
    fig.text(0.5, -0.06, "reverse (learned):  denoise, noise $\\rightarrow$ data",
             ha="center", color=INK, fontsize=11)
    _save(fig, "forward_reverse_moons.jpg")


# ---------------------------------------------------------------- Fig 2
def fig_noise_schedule():
    T = 1000
    beta = np.linspace(1e-4, 0.02, T)               # linear schedule (Ho et al.)
    alpha = 1 - beta
    abar = np.cumprod(alpha)
    t = np.linspace(0, 1, T)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    _style(axL)
    axL.plot(t, beta, color=BLUE, lw=2.2)
    axL.set_title("Per-step noise $\\beta_t$ (small, increasing)", color=INK, fontsize=10.5)
    axL.set_xlabel("time  $t$  (0 = data, 1 = noise)", color=INK2)
    axL.set_ylabel("$\\beta_t$", color=INK2)

    _style(axR)
    axR.plot(t, abar, color=INK2, lw=2.0, ls="--", label="$\\bar\\alpha_t$  (signal power)")
    axR.plot(t, np.sqrt(abar), color=BLUE, lw=2.2, label="$\\sqrt{\\bar\\alpha_t}$  (signal coeff)")
    axR.plot(t, np.sqrt(1 - abar), color=AQUA, lw=2.2, label="$\\sqrt{1-\\bar\\alpha_t}$  (noise coeff)")
    axR.set_title("Signal fades, noise grows  ($\\bar\\alpha_t \\to 0$)", color=INK, fontsize=10.5)
    axR.set_xlabel("time  $t$", color=INK2)
    axR.set_ylabel("coefficient", color=INK2)
    axR.legend(loc="center right", fontsize=9, frameon=False)
    _save(fig, "noise_schedule.jpg")


# ---------------------------------------------------------------- Fig 3
def fig_rectified_crossings():
    rng = np.random.default_rng(0)
    n = 8
    y_noise = np.linspace(-1.6, 1.6, n) + rng.normal(0, 0.08, n)
    y_data = np.linspace(-1.4, 1.4, n)[::-1] + rng.normal(0, 0.08, n)  # reversed -> crossings
    x0 = np.stack([np.full(n, -2.0), y_noise], 1)     # noise samples (pi_0)
    x1 = np.stack([np.full(n, 2.0), y_data], 1)       # data samples (pi_1)
    disp = x1 - x0

    # kernel (Nadaraya-Watson) estimate of the marginal velocity
    #   v(x,t) = E[x1 - x0 | x_t = x]  ~  sum_i w_i (x1_i - x0_i),  w_i ~ exp(-||x-x_t^i||^2/2h^2)
    h = 0.55
    def vfield(x, t):
        xt = (1 - t) * x0 + t * x1                     # (n,2) interpolation points at time t
        d2 = ((x[None, :] - xt) ** 2).sum(1)
        w = np.exp(-d2 / (2 * h * h)); w /= w.sum() + 1e-12
        return (w[:, None] * disp).sum(0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: random straight-line pairings cross
    _style(axL, grid=False)
    for i in range(n):
        axL.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]],
                 color=INK2, lw=1.0, alpha=0.55, zorder=1)
    axL.scatter(x0[:, 0], x0[:, 1], color=AQUA, s=45, zorder=3, label="noise  $\\pi_0$")
    axL.scatter(x1[:, 0], x1[:, 1], color=BLUE, s=45, zorder=3, label="data  $\\pi_1$")
    axL.set_title("Random pairing: straight reference lines CROSS", color=INK, fontsize=10.5)
    axL.set_aspect("equal"); axL.set_xticks([]); axL.set_yticks([])
    axL.legend(loc="upper center", fontsize=9, frameon=False, ncol=2)

    # right: integral curves of the marginal velocity -> non-crossing, curved
    _style(axR, grid=False)
    steps = 300; dt = 1.0 / steps
    for j in range(n):
        z = x0[j].copy(); traj = [z.copy()]
        t = 0.0
        for _ in range(steps):
            z = z + dt * vfield(z, t); t += dt
            traj.append(z.copy())
        traj = np.array(traj)
        axR.plot(traj[:, 0], traj[:, 1], color=BLUE, lw=1.6, alpha=0.9, zorder=2)
    axR.scatter(x0[:, 0], x0[:, 1], color=AQUA, s=45, zorder=3)
    axR.scatter(x1[:, 0], x1[:, 1], color=BLUE, s=45, zorder=3)
    axR.set_title("Rectified marginal flow: averaged, non-crossing (curved)", color=INK, fontsize=10.5)
    axR.set_aspect("equal"); axR.set_xticks([]); axR.set_yticks([])
    _save(fig, "rectified_crossings.jpg")


# ---------------------------------------------------------------- Fig 4
def fig_sde_vs_ode():
    z0 = np.array([2.2, 1.6])
    theta = 2.2                     # pull toward origin
    N = 400; dt = 1.0 / N

    # deterministic ODE: dz = -theta z dt
    z = z0.copy(); ode = [z.copy()]
    for _ in range(N):
        z = z - theta * z * dt
        ode.append(z.copy())
    ode = np.array(ode)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _style(ax, grid=False)
    # several stochastic SDE runs from the same start
    sigma = 0.85
    for s in range(6):
        rng = np.random.default_rng(s + 10)
        z = z0.copy(); sde = [z.copy()]
        for _ in range(N):
            z = z - theta * z * dt + sigma * np.sqrt(dt) * rng.normal(0, 1, 2)
            sde.append(z.copy())
        sde = np.array(sde)
        ax.plot(sde[:, 0], sde[:, 1], color=AQUA, lw=1.0, alpha=0.5,
                label="SDE runs (stochastic)" if s == 0 else None, zorder=2)
    ax.plot(ode[:, 0], ode[:, 1], color=BLUE, lw=2.6, label="ODE (deterministic)", zorder=3)
    ax.scatter([z0[0]], [z0[1]], color=INK, s=55, zorder=4)
    ax.annotate("same start", z0, xytext=(z0[0] - 1.3, z0[1] + 0.2), color=INK, fontsize=9.5)
    ax.scatter([0], [0], color=INK, s=30, zorder=4)
    ax.annotate("target region", (0, 0), xytext=(0.15, -0.9), color=INK2, fontsize=9)
    ax.set_title("Deterministic ODE (one smooth path) vs stochastic SDE\n"
                 "(a different jittery path every run)", color=INK, fontsize=10.5)
    ax.set_xlabel("dimension 1", color=INK2); ax.set_ylabel("dimension 2", color=INK2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    _save(fig, "sde_vs_ode.jpg")


if __name__ == "__main__":
    fig_forward_reverse()
    fig_noise_schedule()
    fig_rectified_crossings()
    fig_sde_vs_ode()
    print("done")
