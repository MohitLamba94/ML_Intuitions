"""
Jensen's inequality figure for `ddpm_ddim_flow_score.md` (Extras -> Probability Primer).

For a concave function (here log), the chord between two points lies BELOW the curve,
so  log E[Y] >= E[log Y].  We take Y in {y1, y2} with equal probability, so:
    E[Y]      = (y1 + y2)/2
    log E[Y]  = value on the CURVE at E[Y]            (upper dot)
    E[log Y]  = (log y1 + log y2)/2 = midpoint of the CHORD  (lower dot)

Run:
    /Users/mohitl/Documents/miniconda3/envs/general/bin/python scripts/jensen_plot.py
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


def main():
    y1, y2 = 0.7, 6.5
    x = np.linspace(0.35, 7.3, 400)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    # concave curve y = log x
    ax.plot(x, np.log(x), color=BLUE, lw=2.4, label="$\\log x$  (concave)")
    # chord between the two points
    ax.plot([y1, y2], [np.log(y1), np.log(y2)], color=AQUA, lw=2.2, ls="-",
            label="chord")

    EY = (y1 + y2) / 2
    logEY = np.log(EY)          # on the curve
    ElogY = (np.log(y1) + np.log(y2)) / 2   # midpoint of chord

    # guide line at E[Y]
    ax.plot([EY, EY], [ElogY, logEY], color=INK2, lw=1.0, ls=":")
    ax.scatter([EY], [logEY], color=BLUE, s=55, zorder=5)
    ax.scatter([EY], [ElogY], color=AQUA, s=55, zorder=5)
    # endpoints
    ax.scatter([y1, y2], [np.log(y1), np.log(y2)], color=INK, s=30, zorder=5)

    ax.annotate("$\\log \\mathbb{E}[Y]$  (on the curve)", (EY, logEY),
                xytext=(EY + 0.2, logEY + 0.35), color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax.annotate("$\\mathbb{E}[\\log Y]$  (on the chord)", (EY, ElogY),
                xytext=(EY + 0.25, ElogY - 0.75), color="#0f7a55", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=0.9))
    ax.annotate("$y_1$", (y1, np.log(y1)), xytext=(y1 - 0.1, np.log(y1) + 0.25),
                color=INK2, fontsize=9)
    ax.annotate("$y_2$", (y2, np.log(y2)), xytext=(y2 - 0.1, np.log(y2) + 0.2),
                color=INK2, fontsize=9)
    ax.annotate("$\\mathbb{E}[Y]$", (EY, np.log(0.36)), xytext=(EY - 0.25, np.log(0.36)),
                color=INK2, fontsize=9)

    ax.set_title("Jensen's inequality:  $\\log \\mathbb{E}[Y] \\geq \\mathbb{E}[\\log Y]$\n"
                 "(the chord of a concave curve lies below it)", color=INK, fontsize=10.5)
    ax.set_xlabel("$y$", color=INK2)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.patch.set_facecolor(SURFACE)
    p = os.path.join(ASSETS, "jensen_inequality.jpg")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.normpath(p))


if __name__ == "__main__":
    main()
