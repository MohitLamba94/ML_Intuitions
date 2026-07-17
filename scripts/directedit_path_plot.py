"""
Schematic for `training-free-image-editing.md` (the DirectEdit note).

Builds intuition for WHY inversion accumulates error and how DirectEdit's
residual injection fixes it, using a 1-D toy where the vertical axis is a
scalar stand-in for the latent state Z and the horizontal axis is the
denoising step.

  Left  (Vanilla Euler): the reconstruction path evaluates the velocity at the
        WRONG (current) state, so every step leaves a small gap; the gaps are
        the same sign and COMPOUND, and the reconstruction peels away from the
        inversion trajectory -> large final reconstruction error.
  Right (DirectEdit): the recorded inversion residual is injected before each
        velocity call, snapping the reconstruction back onto the inversion
        trajectory at every step -> the two paths coincide (Error ~ 0).

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python \
        scripts/directedit_path_plot.py

Saves assets/directedit_path_alignment.jpg. Palette follows the repo's other
plot scripts (dataviz categorical slots on a light surface).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INV = "#2a78d6"     # inversion trajectory (blue, the reference)
REC = "#e0562d"     # reconstruction / editing path (warm)
OK = "#1baf7a"      # aligned / good
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"
ERR = "#c0392b"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

N = 8                       # number of Euler steps shown
t = np.linspace(0.0, 1.0, N + 1)     # flow time: t=0 noise, t=1 image
# The (shared) inversion trajectory, in the note's notation Z_t = t*Z_1+(1-t)*Z_0.
# Using scalar proxies Z_0 = 0 (noise), Z_1 = 1 (image), so the ideal state is t.
inv = t.copy()


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(title, color=INK, fontsize=12, pad=10, weight="bold")
    ax.set_xlabel("flow time  $t$   ($t=0$ noise  →  $t=1$ image)",
                  color=INK2, fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.16)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["$Z_0$\n(noise)", "$Z_1$\n(image)"], color=INK2,
                       fontsize=9)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def _dir_arrow(ax, x0, x1, yfunc, color, label, dy=0.0, va="bottom"):
    """Draw a progression arrow from t=x0 to t=x1 along a curve yfunc(t)."""
    ax.annotate("", xy=(x1, yfunc(x1) + dy), xytext=(x0, yfunc(x0) + dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                mutation_scale=18))
    xm = 0.5 * (x0 + x1)
    ax.text(xm, yfunc(xm) + dy, label, color=color, fontsize=8.5,
            weight="bold", ha="center", va=va, rotation=38,
            rotation_mode="anchor")


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))
line = lambda x: x                                    # ideal state = t

# ----------------------------------------------------------------- LEFT: Euler
_style(axL, "(a) Vanilla Euler inversion — errors compound")
# inversion trajectory (image -> noise), the reference we WANT to retrace
axL.plot(t, inv, color=INV, lw=2.6, marker="o", ms=6,
         label="inversion path (reference)", zorder=3)
# reconstruction path (noise -> image): each Euler step drifts by the same sign,
# so the gap to the reference compounds and never reaches the image.
rec = np.empty(N + 1)
rec[0] = inv[0]                       # start together at the noise end (t=0)
drift = 0.03
for k in range(1, N + 1):
    ideal_step = inv[k] - inv[k - 1]            # the ascent one step should make
    rec[k] = rec[k - 1] + ideal_step - drift    # under-shoots -> gap grows
axL.plot(t, rec, color=REC, lw=2.6, marker="s", ms=6, ls="--",
         label="reconstruction path", zorder=3)
# per-step error bars (vertical gaps)
for k in range(1, N + 1):
    axL.plot([t[k], t[k]], [inv[k], rec[k]], color=ERR, lw=1.4, alpha=0.8,
             zorder=2)
# progression arrows: inversion runs image->noise (decreasing t),
# reconstruction runs noise->image (increasing t)
_dir_arrow(axL, 0.82, 0.60, line, INV, "image → noise", dy=0.045, va="bottom")
_dir_arrow(axL, 0.18, 0.44, lambda x: np.interp(x, t, rec), REC,
           "noise → image", dy=-0.10, va="top")
axL.annotate("gap grows\nevery step", xy=(t[6], (inv[6] + rec[6]) / 2),
             xytext=(0.66, 0.30), color=ERR, fontsize=10, weight="bold",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=ERR, lw=1.3))
axL.annotate("never reaches\nthe image", xy=(t[N], rec[N]), xytext=(0.55, 0.98),
             color=ERR, fontsize=10, weight="bold", ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=ERR, lw=1.3))
axL.legend(loc="lower right", fontsize=9, frameon=False)

# ------------------------------------------------------------ RIGHT: DirectEdit
_style(axR, "(b) DirectEdit — residual injection re-aligns each step")
axR.plot(t, inv, color=INV, lw=3.4, marker="o", ms=6,
         label="inversion path (reference)", zorder=2)
# reconstruction coincides with inversion; drawn dashed on top for visibility
axR.plot(t, inv, color=OK, lw=2.0, marker="s", ms=5, ls="--",
         label="reconstruction path (aligned)", zorder=3)
# progression arrow: reconstruction rides the reference noise->image
_dir_arrow(axR, 0.20, 0.46, line, OK, "noise → image", dy=-0.10, va="top")
# little magenta residual arrows showing the injected correction at a few steps
for k in [2, 4, 6]:
    axR.annotate("", xy=(t[k], inv[k]), xytext=(t[k] + 0.07, inv[k] + 0.14),
                 arrowprops=dict(arrowstyle="->", color="#b5179e", lw=1.6))
axR.text(t[6] + 0.09, inv[6] + 0.11, r"$+\,\Delta Z_t$" + "\n(recorded\nresidual)",
         color="#b5179e", fontsize=10, weight="bold", ha="left", va="center")
axR.annotate("Error ≈ 0\n(paths coincide)", xy=(t[3], inv[3]),
             xytext=(0.60, 0.30), color=OK, fontsize=11, weight="bold",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=OK, lw=1.3))
axR.legend(loc="upper left", fontsize=9, frameon=False)

fig.suptitle(
    "Reconstruction vs. inversion trajectory: why alignment matters",
    color=INK, fontsize=13.5, weight="bold", y=1.005)
fig.patch.set_facecolor(SURFACE)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(ASSETS, "directedit_path_alignment.jpg")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("wrote", os.path.normpath(out))
